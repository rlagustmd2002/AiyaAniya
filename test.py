# -*- coding: utf-8 -*-
import os
import csv
import json
import shutil
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import train as T

MAX_COPY_PER_TYPE = 20 # 발표 시연용으로 복사할 오디오 최대 개수 (FP/FN 각각)
THRESHOLD = 0.5


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[threshold] {THRESHOLD}")

    # 1. 데이터 분할 불러오기
    split_path = os.path.join(T.RESULTS_DIR, "dataset_split.json")
    if not os.path.exists(split_path):
        for cand in ["results/dataset_split.json", "./dataset_split.json"]:
            if os.path.exists(cand):
                split_path = cand
                break
    print(f"[split] {split_path}")
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)
    test_speakers = split["test_speakers"]
    print(f"[test] 화자 {len(test_speakers)}명")

    # 2. 테스트 항목 구성
    print("[build] train 코드의 collect_speaker_data() 재사용")
    speaker_data = T.collect_speaker_data()   # {speaker: [(real_path, fake_path), ...]}

    test_set = set(test_speakers)
    test_items = []
    for speaker, pairs in speaker_data.items():
        if speaker not in test_set:
            continue
        for real_path, fake_path in pairs:
            test_items.append((real_path, 0))   # real = 0
            test_items.append((fake_path, 1))   # fake = 1

    print(f"[test] 화자 {len(test_set & set(speaker_data.keys()))}명 / "
          f"샘플 {len(test_items)}개")
    if len(test_items) == 0:
        raise RuntimeError(
            "테스트 샘플이 0개입니다. collect_speaker_data()의 반환 형식이 "
            "{speaker: [(real,fake),...]} 가 맞는지, test_speakers 이름이 "
            "폴더명과 일치하는지 확인하세요.")

    # 3. 모델 로드 (best_model.pth)
    ckpt_path = os.path.join(T.CHECKPOINT_DIR, "best_model.pth")
    print(f"[model] {ckpt_path} 로드 중...")
    model = T.DeepfakeDetector().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    # 4. 테스트셋 추론 (augment=False)
    test_ds = T.DeepfakeDataset(test_items, augment=False)
    test_loader = DataLoader(test_ds, batch_size=getattr(T, "BATCH_SIZE", 8),
                             shuffle=False, num_workers=0)

    rows = []   # (path, true_label, prob, pred_label, result)
    idx = 0
    with torch.no_grad():
        for waveforms, labels in tqdm(test_loader, desc="추론"):
            waveforms = waveforms.to(device)
            logits = model(waveforms).squeeze(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            for p in probs:
                path, true = test_items[idx]
                pred = 1 if p >= THRESHOLD else 0
                if true == 0 and pred == 0:   result = "TN"
                elif true == 0 and pred == 1: result = "FP"
                elif true == 1 and pred == 0: result = "FN"
                else:                          result = "TP"
                rows.append((path, true, float(p), pred, result))
                idx += 1

    # 5. 결과 저장
    out_dir = T.RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)

    def write_csv(path, data):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["file_path", "true_label(0=real,1=fake)",
                        "fake_prob", "pred_label", "result"])
            w.writerows(data)

    all_csv = os.path.join(out_dir, "error_analysis.csv")
    fp_csv  = os.path.join(out_dir, "errors_FP.csv")
    fn_csv  = os.path.join(out_dir, "errors_FN.csv")
    fps = [r for r in rows if r[4] == "FP"]
    fns = [r for r in rows if r[4] == "FN"]

    write_csv(all_csv, rows)
    write_csv(fp_csv, fps)
    write_csv(fn_csv, fns)

    # 6. 통계 출력
    n = len(rows)
    tn = sum(1 for r in rows if r[4] == "TN")
    tp = sum(1 for r in rows if r[4] == "TP")
    fp = len(fps); fn = len(fns)
    acc = (tn + tp) / n if n else 0
    print("\n" + "=" * 50)
    print(f"  전체 테스트 샘플: {n}")
    print(f"  TN(진짜 정답): {tn}   FP(진짜→가짜 오판): {fp}")
    print(f"  FN(가짜→진짜 오판): {fn}   TP(가짜 정답): {tp}")
    print(f"  정확도: {acc*100:.2f}%")
    print("=" * 50)
    print(f"  전체 결과: {all_csv}")
    print(f"  FP 목록:   {fp_csv} ({fp}개)")
    print(f"  FN 목록:   {fn_csv} ({fn}개)")

    # 7. 오탐 오디오 복사 (발표 시연용)
    sample_root = os.path.join(out_dir, "error_samples")
    fp_sorted = sorted(fps, key=lambda r: -r[2])[:MAX_COPY_PER_TYPE]
    fn_sorted = sorted(fns, key=lambda r: r[2])[:MAX_COPY_PER_TYPE]

    for kind, data in [("FP", fp_sorted), ("FN", fn_sorted)]:
        dst_dir = os.path.join(sample_root, kind)
        os.makedirs(dst_dir, exist_ok=True)
        for i, (path, true, prob, pred, result) in enumerate(data, 1):
            if os.path.exists(path):
                base = os.path.basename(path)
                dst = os.path.join(dst_dir, f"{i:02d}_prob{prob:.3f}_{base}")
                shutil.copy2(path, dst)
    print(f"  오탐 오디오 복사: {sample_root}/FP, {sample_root}/FN "
          f"(각 최대 {MAX_COPY_PER_TYPE}개)")
    print("\n완료!")


if __name__ == "__main__":
    main()