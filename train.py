import os
import csv
import json
import glob
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    confusion_matrix, roc_curve, precision_score, recall_score
)
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import Wav2Vec2Model
from tqdm import tqdm

# 데이터 경로
BASE_DATASET_DIR = r"E:\Project\AiyaAniya\datasets"
ORIGIN_VOICE_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice")
FAKE_VOICE_DIR   = os.path.join(BASE_DATASET_DIR, "fake_voice")
LIST_DIR         = os.path.join(BASE_DATASET_DIR, "dataset_list")

# 출력 경로
_BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(_BASE_DIR, "checkpoints")
RESULTS_DIR    = os.path.join(_BASE_DIR, "results")

# 학습 파라미터
BATCH_SIZE    = 8
EPOCHS        = 50
LEARNING_RATE = 1e-5
WEIGHT_DECAY  = 1e-4
SAMPLE_RATE   = 16000
MAX_LEN       = 16000 * 4
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15  
# TEST_RATIO  = 0.15

# 화자당 최대 사용 쌍 수 (real/fake 각각 이만큼)
MAX_SAMPLES_PER_SPEAKER = 2000

# 도메인 통일 / 증강 설정
MATCH_BANDWIDTH   = False
COMMON_LOWPASS_HZ = 7000
AUG_BANDWIDTH     = True
AUG_SILENCE_NOISE = True

# Early Stopping
EARLY_STOPPING_PATIENCE = 7

# unfreeze할 wav2vec2 상위 트랜스포머 레이어 수
UNFREEZE_LAST_N_LAYERS = 4

NUM_WORKERS = 4
USE_AMP = True
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def log(msg, level="INFO"):
    tag = {"INFO": "[ INFO ]", "OK": "[  OK  ]",
           "WARN": "[ WARN ]", "ERROR": "[ERROR ]"}.get(level, "[ INFO ]")
    print(f"{tag} {msg}", flush=True)


def _lowpass(waveform, sr, cutoff):
    cutoff = max(500.0, min(float(cutoff), sr / 2.0 - 100.0))
    return torchaudio.functional.lowpass_biquad(waveform, sr, cutoff)

# 데이터 수집 (화자 단위)
def collect_speaker_data():
    speaker_data = {}

    speaker_folders = sorted([
        d for d in os.listdir(ORIGIN_VOICE_DIR)
        if os.path.isdir(os.path.join(ORIGIN_VOICE_DIR, d))
    ])

    log(f"총 {len(speaker_folders)}명 화자 탐색 중...")

    for speaker in speaker_folders:
        list_path     = os.path.join(LIST_DIR, f"{speaker}.list")
        fake_dir      = os.path.join(FAKE_VOICE_DIR, speaker)
        real_voice_dir = os.path.join(ORIGIN_VOICE_DIR, speaker)

        if not os.path.exists(list_path):
            log(f"  [SKIP] {speaker}: .list 파일 없음", "WARN")
            continue
        if not os.path.isdir(fake_dir):
            log(f"  [SKIP] {speaker}: fake_voice 폴더 없음", "WARN")
            continue
        if not os.path.isdir(real_voice_dir):
            log(f"  [SKIP] {speaker}: origin_voice 폴더 없음", "WARN")
            continue

        # fake_voice 폴더가 비어있으면 스킵
        fake_count = len(glob.glob(os.path.join(fake_dir, "*.wav")))
        if fake_count == 0:
            log(f"  [SKIP] {speaker}: 가짜 음성 없음", "WARN")
            continue

        pairs = []
        with open(list_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(pairs) >= MAX_SAMPLES_PER_SPEAKER:
                    break
                parts = line.strip().split("|")
                if len(parts) < 1:
                    continue

                old_real_path = parts[0].strip().replace("\\", "/")
                real_filename = os.path.basename(old_real_path)
                real_path     = os.path.join(real_voice_dir, real_filename)

                fake_name = f"{speaker}_FAKE_{i+1:06d}.wav"
                fake_path = os.path.join(fake_dir, fake_name)

                if os.path.exists(real_path) and os.path.exists(fake_path):
                    pairs.append((real_path, fake_path))

        if len(pairs) == 0:
            log(f"  [SKIP] {speaker}: 유효한 쌍 없음", "WARN")
            continue

        speaker_data[speaker] = pairs
        log(f"  [OK] {speaker}: {len(pairs)}쌍")

    return speaker_data


def split_speakers(speaker_data, seed=SEED):
    speakers = list(speaker_data.keys())
    random.Random(seed).shuffle(speakers)

    n = len(speakers)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    train_speakers = speakers[:n_train]
    val_speakers   = speakers[n_train:n_train + n_val]
    test_speakers  = speakers[n_train + n_val:]

    log(f"\n[화자 분할]")
    log(f"  Train ({len(train_speakers)}명): {train_speakers}")
    log(f"  Val   ({len(val_speakers)}명): {val_speakers}")
    log(f"  Test  ({len(test_speakers)}명): {test_speakers}")

    def flatten(spk_list):
        items = []
        for spk in spk_list:
            for real, fake in speaker_data[spk]:
                items.append((real, 0))   # real=0
                items.append((fake, 1))   # fake=1
        return items

    return (flatten(train_speakers), flatten(val_speakers), flatten(test_speakers),
            train_speakers, val_speakers, test_speakers)

#  Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, items, max_len=MAX_LEN, augment=False):
        """
        items: [(audio_path, label), ...]   label: 0=real, 1=fake
        augment: train 시 True
        """
        self.items   = items
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        audio_path, label = self.items[idx]

        try:
            waveform, sr = torchaudio.load(audio_path)
        except Exception as e:
            print(f"\n[로드 실패] {audio_path}: {e}")
            waveform = torch.zeros(1, self.max_len)
            sr = SAMPLE_RATE

        if sr != SAMPLE_RATE:
            waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # 도메인 통일: real(48k)/fake(32k) 파이프라인 차이 제거
        if MATCH_BANDWIDTH:
            waveform = _lowpass(waveform, SAMPLE_RATE, COMMON_LOWPASS_HZ)

        if waveform.size(0) > self.max_len:
            if self.augment:
                start = random.randint(0, waveform.size(0) - self.max_len)
                waveform = waveform[start:start + self.max_len]
            else:
                start = (waveform.size(0) - self.max_len) // 2
                waveform = waveform[start:start + self.max_len]
        else:
            pad = torch.zeros(self.max_len - waveform.size(0))
            waveform = torch.cat((waveform, pad), dim=0)

        # 정규화
        mean = waveform.mean()
        std  = waveform.std() + 1e-7
        waveform = (waveform - mean) / std

        if self.augment:
            # 대역폭 랜덤화
            if AUG_BANDWIDTH and random.random() < 0.5:
                waveform = _lowpass(waveform, SAMPLE_RATE, random.uniform(3500, 7500))

            if random.random() < 0.3:
                noise = torch.randn_like(waveform) * 0.005
                waveform = waveform + noise
            if random.random() < 0.3:
                gain = random.uniform(0.8, 1.2)
                waveform = waveform * gain
            if AUG_SILENCE_NOISE and random.random() < 0.5:
                waveform = waveform + torch.randn_like(waveform) * random.uniform(0.002, 0.01)

        return waveform, torch.tensor(label, dtype=torch.float32)

#  모델
class DeepfakeDetector(nn.Module):
    def __init__(self, unfreeze_last_n=UNFREEZE_LAST_N_LAYERS):
        super().__init__()

        log("  wav2vec2-base 로딩 중...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base", use_safetensors=True)

        # 전체 freeze
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        # 상위 N개 트랜스포머 레이어 unfreeze
        total_layers = len(self.wav2vec2.encoder.layers)
        for layer in self.wav2vec2.encoder.layers[total_layers - unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # feature projection도 unfreeze
        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True

        # 학습 대상 파라미터 수 출력
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        log(f"  학습 파라미터: {trainable:,} / 전체: {total:,} ({trainable/total*100:.1f}%)")

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        outputs = self.wav2vec2(x)
        hidden  = outputs.last_hidden_state  # (B, T, 768)

        # mean + max pooling 결합
        mean_pool = hidden.mean(dim=1)
        max_pool  = hidden.max(dim=1).values
        pooled    = (mean_pool + max_pool) / 2

        return self.classifier(pooled).squeeze(-1)

#  그래프 저장 함수
def save_training_curve(history, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Training Curves (Epoch {epoch})", fontsize=14)

    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs_range, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs_range, history["val_loss"],   "r-o", label="Val Loss",   markersize=4)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs_range, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs_range, history["val_acc"],   "r-o", label="Val Acc",   markersize=4)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC
    axes[2].plot(epochs_range, history["val_auc"], "g-o", label="Val AUC", markersize=4)
    axes[2].set_title("Validation AUC-ROC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    latest_path = os.path.join(save_dir, "training_curves_latest.png")
    plt.savefig(latest_path, dpi=150, bbox_inches="tight")

    # 5에폭마다 백업
    if epoch % 5 == 0 or epoch == 1:
        backup_path = os.path.join(save_dir, f"training_curves_e{epoch:03d}.png")
        plt.savefig(backup_path, dpi=150, bbox_inches="tight")

    plt.close()


def save_final_evaluation(y_true, y_prob, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    y_pred = (np.array(y_prob) >= 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Final Test Evaluation", fontsize=14)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    axes[0].plot(fpr, tpr, "b-", lw=2, label=f"AUC = {auc:.4f}")
    axes[0].plot([0, 1], [0, 1], "r--", lw=1)
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    im = axes[1].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].set_title("Confusion Matrix")
    plt.colorbar(im, ax=axes[1])
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(["Real (0)", "Fake (1)"])
    axes[1].set_yticklabels(["Real (0)", "Fake (1)"])
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=14)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "final_evaluation.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # 지표 계산
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)

    metrics = {
        "accuracy":  float(acc),
        "f1_score":  float(f1),
        "precision": float(precision),
        "recall":    float(recall),
        "auc_roc":   float(auc),
        "confusion_matrix": cm.tolist(),
    }

    # JSON으로 저장
    with open(os.path.join(save_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  ★ 최종 테스트 결과")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print(f"  Confusion Matrix:\n{cm}")

    return metrics

#  학습 / 평가 루프
def run_epoch(model, loader, criterion, optimizer=None, scaler=None, desc=""):
    """한 에폭 실행. optimizer=None이면 평가."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    all_probs  = []
    all_preds  = []
    all_labels = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for waveform, label in tqdm(loader, desc=desc, leave=False):
            waveform = waveform.to(DEVICE, non_blocking=True)
            label    = label.to(DEVICE, non_blocking=True)

            if is_train:
                optimizer.zero_grad()

            if USE_AMP and DEVICE.type == "cuda":
                with autocast():
                    output = model(waveform)
                    loss   = criterion(output, label)

                if is_train:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                output = model(waveform)
                loss   = criterion(output, label)

                if is_train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item()

            probs  = torch.sigmoid(output).detach().cpu().numpy()
            preds  = (probs >= 0.5).astype(int)
            labels = label.detach().cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0

    return avg_loss, acc, auc, all_probs, all_labels

#  메인
def train():
    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"VRAM: {vram_gb:.1f} GB")
        log(f"Mixed Precision (AMP): {'사용' if USE_AMP else '미사용'}")

    # 데이터 수집 / 분할
    speaker_data = collect_speaker_data()

    (train_items, val_items, test_items,
     train_speakers, val_speakers, test_speakers) = split_speakers(speaker_data)

    # 분할 정보 저장
    with open(os.path.join(RESULTS_DIR, "dataset_split.json"), "w", encoding="utf-8") as f:
        json.dump({
            "train_speakers": train_speakers,
            "val_speakers":   val_speakers,
            "test_speakers":  test_speakers,
            "train_size":     len(train_items),
            "val_size":       len(val_items),
            "test_size":      len(test_items),
        }, f, indent=2, ensure_ascii=False)

    # 클래스 균형 출력
    def count_labels(items):
        real = sum(1 for _, l in items if l == 0)
        fake = sum(1 for _, l in items if l == 1)
        return real, fake

    tr_real, tr_fake = count_labels(train_items)
    vl_real, vl_fake = count_labels(val_items)
    ts_real, ts_fake = count_labels(test_items)

    log(f"\n[데이터셋 크기]")
    log(f"  Train: {len(train_items):,}개 (real {tr_real:,} / fake {tr_fake:,})")
    log(f"  Val  : {len(val_items):,}개 (real {vl_real:,} / fake {vl_fake:,})")
    log(f"  Test : {len(test_items):,}개 (real {ts_real:,} / fake {ts_fake:,})")
    log(f"  합계 : {len(train_items)+len(val_items)+len(test_items):,}개")

    train_dataset = DeepfakeDataset(train_items, augment=True)
    val_dataset   = DeepfakeDataset(val_items,   augment=False)
    test_dataset  = DeepfakeDataset(test_items,  augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    log("\n[모델 초기화]")
    model = DeepfakeDetector().to(DEVICE)
    unfreeze_params   = []
    classifier_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in name:
            classifier_params.append(param)
        else:
            unfreeze_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": unfreeze_params,   "lr": LEARNING_RATE},
        {"params": classifier_params, "lr": LEARNING_RATE * 10},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-7
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler    = GradScaler() if (USE_AMP and DEVICE.type == "cuda") else None

    # CSV 로그
    log_path = os.path.join(RESULTS_DIR, "training_log.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "train_auc",
                         "val_loss", "val_acc", "val_auc", "lr"])

    # 학습
    history = {
        "train_loss": [], "train_acc": [], "train_auc": [],
        "val_loss":   [], "val_acc":   [], "val_auc":   [],
    }

    best_val_loss    = float("inf")
    patience_counter = 0
    best_epoch       = 0

    log(f"\n[학습 시작] Epochs: {EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[Epoch {epoch}/{EPOCHS}]")

        tr_loss, tr_acc, tr_auc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            desc=f"  Train e{epoch}"
        )
        vl_loss, vl_acc, vl_auc, _, _ = run_epoch(
            model, val_loader, criterion, scaler=scaler,
            desc=f"  Val   e{epoch}"
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["train_auc"].append(tr_auc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_auc"].append(vl_auc)

        print(f"  Train | Loss: {tr_loss:.4f}  Acc: {tr_acc:.4f}  AUC: {tr_auc:.4f}")
        print(f"  Val   | Loss: {vl_loss:.4f}  Acc: {vl_acc:.4f}  AUC: {vl_auc:.4f}")
        print(f"  LR: {current_lr:.2e}")

        # CSV 로그
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, tr_loss, tr_acc, tr_auc,
                             vl_loss, vl_acc, vl_auc, current_lr])

        # 그래프 저장
        save_training_curve(history, epoch, RESULTS_DIR)

        # 최고 모델 저장 (추론용 메타정보 포함)
        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            best_epoch       = epoch
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":            vl_loss,
                "val_acc":             vl_acc,
                "val_auc":             vl_auc,
                # 추론 시 필요한 메타정보
                "config": {
                    "sample_rate":      SAMPLE_RATE,
                    "max_len":          MAX_LEN,
                    "wav2vec2_model":   "facebook/wav2vec2-base",
                    "unfreeze_last_n":  UNFREEZE_LAST_N_LAYERS,
                    "label_mapping":    {"0": "real", "1": "fake"},
                },
            }, os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ★ 최고 모델 저장 (val_loss: {vl_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  Early Stopping 카운터: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        # 5 에폭마다 체크포인트
        if epoch % 5 == 0:
            torch.save({
                "epoch":               epoch,
                "model_state_dict":    model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":            vl_loss,
            }, os.path.join(CHECKPOINT_DIR, f"checkpoint_e{epoch:03d}.pth"))

        # Early Stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n  Early Stopping! ({EARLY_STOPPING_PATIENCE} 에폭 동안 개선 없음)")
            print(f"  최고 성능: Epoch {best_epoch}, val_loss={best_val_loss:.4f}")
            break

    # 최종 테스트
    print(f"\n[최종 테스트 평가] best_model.pth (epoch {best_epoch}) 로드...")
    best_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "best_model.pth"),
        map_location=DEVICE
    )
    model.load_state_dict(best_ckpt["model_state_dict"])

    _, _, _, test_probs, test_labels = run_epoch(
        model, test_loader, criterion, scaler=scaler,
        desc="  Test"
    )

    save_final_evaluation(test_labels, test_probs, RESULTS_DIR)
    print(f"\n학습 완료! 결과 저장 위치: {RESULTS_DIR}")
    print(f"최종 모델 경로: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    train()