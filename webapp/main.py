import os
import io
import time
import shutil
import subprocess
import tempfile
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from transformers import Wav2Vec2Model

MODEL_PATH = r"E:\Project\AiyaAniya\checkpoints\best_model.pth"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
PORT = 8000

FFMPEG_PATH = r"E:\GPT-SoVITS-v3lora-20250228\ffmpeg.exe"
if not os.path.exists(FFMPEG_PATH):
    FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"

FAKE_THRESHOLD = 0.5
MIN_AUDIO_RMS = 0.01
COMMON_LOWPASS_HZ = 7000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepfakeDetector(nn.Module):
    def __init__(self, unfreeze_last_n=4):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base",
            use_safetensors=True,
        )

        for param in self.wav2vec2.parameters():
            param.requires_grad = False

        total_layers = len(self.wav2vec2.encoder.layers)
        for layer in self.wav2vec2.encoder.layers[total_layers - unfreeze_last_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        for param in self.wav2vec2.feature_projection.parameters():
            param.requires_grad = True

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
        hidden  = outputs.last_hidden_state
        mean_pool = hidden.mean(dim=1)
        max_pool  = hidden.max(dim=1).values
        pooled    = (mean_pool + max_pool) / 2
        return self.classifier(pooled).squeeze(-1)

print("=" * 60)
print("딥보이스 탐지 서버 초기화 중...")
print("=" * 60)
print(f"  Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일 없음: {MODEL_PATH}")

print(f"  모델 로드 중: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

CONFIG = checkpoint.get("config", {
    "sample_rate":    16000,
    "max_len":        16000 * 4,
    "wav2vec2_model": "facebook/wav2vec2-base",
    "unfreeze_last_n": 4,
    "label_mapping":  {"0": "real", "1": "fake"},
})

SAMPLE_RATE = CONFIG["sample_rate"]
MAX_LEN     = CONFIG["max_len"]

model = DeepfakeDetector(unfreeze_last_n=CONFIG["unfreeze_last_n"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

MODEL_INFO = {
    "trained_epoch":  checkpoint.get("epoch", "?"),
    "val_loss":       checkpoint.get("val_loss", None),
    "val_acc":        checkpoint.get("val_acc", None),
    "val_auc":        checkpoint.get("val_auc", None),
    "sample_rate":    SAMPLE_RATE,
    "max_len_sec":    MAX_LEN / SAMPLE_RATE,
    "device":         str(DEVICE),
    "base_model":     CONFIG["wav2vec2_model"],
}

print(f"  모델 로드 완료")
print(f"  학습 epoch: {MODEL_INFO['trained_epoch']}")
print(f"  Val Loss : {MODEL_INFO['val_loss']:.4f}" if MODEL_INFO['val_loss'] else "")
print(f"  Val Acc  : {MODEL_INFO['val_acc']:.4f}" if MODEL_INFO['val_acc'] else "")
print(f"  Val AUC  : {MODEL_INFO['val_auc']:.4f}" if MODEL_INFO['val_auc'] else "")
print("=" * 60)

#  음성 전처리
def preprocess_audio(audio_bytes: bytes) -> tuple:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp_in:
        tmp_in.write(audio_bytes)
        input_path = tmp_in.name
    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-ar", str(SAMPLE_RATE),
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-loglevel", "error",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg 변환 실패: {result.stderr}")
        waveform, sr = torchaudio.load(output_path)
    finally:
        for p in [input_path, output_path]:
            try:
                os.unlink(p)
            except Exception:
                pass
    if sr != SAMPLE_RATE:
        waveform = T.Resample(sr, SAMPLE_RATE)(waveform)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    rms = torch.sqrt(torch.mean(waveform ** 2)).item()
    # waveform = torchaudio.functional.lowpass_biquad(waveform, SAMPLE_RATE, COMMON_LOWPASS_HZ)

    if waveform.size(0) > MAX_LEN:
        start = (waveform.size(0) - MAX_LEN) // 2
        waveform = waveform[start:start + MAX_LEN]
    else:
        if waveform.size(0) > 0:
            reps = (MAX_LEN // waveform.size(0)) + 1
            waveform = waveform.repeat(reps)[:MAX_LEN]
        else:
            waveform = torch.zeros(MAX_LEN)

    mean = waveform.mean()
    std  = waveform.std() + 1e-7
    waveform = (waveform - mean) / std

    return waveform.unsqueeze(0), rms


@torch.no_grad()
def predict_audio(waveform: torch.Tensor) -> dict:
    waveform = waveform.to(DEVICE)
    logit  = model(waveform)
    prob   = torch.sigmoid(logit).item() 

    label = "fake" if prob >= FAKE_THRESHOLD else "real"
    confidence = prob if label == "fake" else (1 - prob)

    return {
        "prediction":       label,
        "confidence":       float(confidence),
        "real_probability": float(1 - prob),
        "fake_probability": float(prob),
    }

app = FastAPI(
    title="딥보이스 탐지 API",
    description="wav2vec2 기반 합성 음성 판별 서버",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """서버 상태 확인."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "model_loaded": True,
    }


@app.get("/model/info")
async def model_info():
    """모델 정보 반환."""
    return MODEL_INFO


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="빈 파일")
    if len(audio_bytes) > 50 * 1024 * 1024:  # 50MB 제한
        raise HTTPException(status_code=400, detail="파일 너무 큼 (50MB 초과)")

    try:
        waveform, rms = preprocess_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"오디오 처리 실패: {e}")

    if rms < MIN_AUDIO_RMS:
        elapsed_ms = int((time.time() - start_time) * 1000)
        return {
            "prediction":         "no_voice",
            "confidence":         0.0,
            "real_probability":   0.0,
            "fake_probability":   0.0,
            "rms":                float(rms),
            "processing_time_ms": elapsed_ms,
            "filename":           file.filename,
            "message":            "음성이 감지되지 않았습니다",
        }

    try:
        result = predict_audio(waveform)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 추론 실패: {e}")

    elapsed_ms = int((time.time() - start_time) * 1000)
    result["processing_time_ms"] = elapsed_ms
    result["rms"]      = float(rms)
    result["filename"] = file.filename

    return result

if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def serve_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "딥보이스 탐지 API 서버 - /docs 에서 API 문서 확인"}
else:
    @app.get("/")
    async def root():
        return {
            "message": "딥보이스 탐지 API 서버",
            "docs":    "/docs",
            "health":  "/health",
        }

if __name__ == "__main__":
    import uvicorn
    print(f"\n서버 시작: http://0.0.0.0:{PORT}")
    print(f"API 문서: http://localhost:{PORT}/docs")
    print(f"웹앱 접속: http://localhost:{PORT}/")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
