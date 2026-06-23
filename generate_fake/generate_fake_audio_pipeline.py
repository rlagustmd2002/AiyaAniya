import os
import re
import glob
import json
import wave
import time
import requests
import traceback
from tqdm import tqdm

# API 서버 주소
API_BASE = "http://127.0.0.1:9880"

# GPT-SoVITS 설치 폴더
GPTSOVITS_DIR = r"E:\GPT-SoVITS-v3lora-20250228"

# 데이터셋 루트
BASE_DATASET_DIR = r"E:\Project\AiyaAniya\datasets"

# 원천 음성 폴더 (ref_audio 탐색용)
ORIGIN_VOICE_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice")

# 라벨링 폴더 (prompt_text 추출용)
ORIGIN_LABEL_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice_labeling")

# .list 파일 폴더 (대본 추출용)
LIST_DIR = os.path.join(BASE_DATASET_DIR, "dataset_list")

# 가짜 음성 출력 폴더
FAKE_VOICE_DIR = os.path.join(BASE_DATASET_DIR, "fake_voice")

# SoVITS 가중치 폴더
SOVITS_WEIGHT_DIR = os.path.join(GPTSOVITS_DIR, "SoVITS_weights_v2ProPlus")

# GPT 가중치 폴더 (두 곳 모두 탐색)
GPT_WEIGHT_DIR_PLUS = os.path.join(GPTSOVITS_DIR, "GPT_weights_v2ProPlus")
GPT_WEIGHT_DIR_PRO  = os.path.join(GPTSOVITS_DIR, "GPT_weights_v2Pro")

# 화자 1명당 생성할 최대 음성 수
MAX_COUNT = 2000

# ref_audio 최소 길이 (초)
MIN_REF_DURATION = 3.0

# 언어
PROMPT_LANG = "ko"
TEXT_LANG   = "ko"

# API 요청 타임아웃 (초)
REQUEST_TIMEOUT = 60

#  내부 유틸 함수
def log(msg, level="INFO"):
    tag = {"INFO": "[ INFO ]", "OK": "[  OK  ]", "WARN": "[ WARN ]", "ERROR": "[ERROR ]"}.get(level, "[ INFO ]")
    print(f"{tag} {msg}", flush=True)

def get_wav_duration(wav_path):
    try:
        with wave.open(wav_path, "r") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def find_ref_audio(wav_dir):
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        return None

    candidates = [(get_wav_duration(w), w) for w in wav_files if get_wav_duration(w) >= MIN_REF_DURATION]
    if candidates:
        candidates.sort(key=lambda x: x[1])  # 파일명 순
        return candidates[0][1]

    all_durs = [(get_wav_duration(w), w) for w in wav_files]
    all_durs.sort(reverse=True)
    return all_durs[0][1]


def get_prompt_text(wav_path, json_dir):
    """ref_audio에 대응하는 JSON에서 TransLabelText 추출."""
    json_name = os.path.splitext(os.path.basename(wav_path))[0] + ".json"
    json_path = os.path.join(json_dir, json_name)
    if not os.path.exists(json_path):
        log(f"  JSON 없음: {json_path}", "WARN")
        return ""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)["전사정보"]["TransLabelText"].strip()
    except Exception as e:
        log(f"  JSON 읽기 실패: {e}", "WARN")
        return ""


def find_best_model(weight_dir, speaker_folder, ext):
    """
    weight_dir에서 speaker_folder로 시작하는 모델 파일 중
    epoch 숫자가 가장 큰 파일 반환.
    없으면 None.
    """
    pattern = os.path.join(weight_dir, f"{speaker_folder}*{ext}")
    files = glob.glob(pattern)
    if not files:
        return None

    def extract_epoch(path):
        # e15, e8, e10 등에서 숫자 추출
        m = re.search(r'[_\-]e(\d+)', os.path.basename(path))
        return int(m.group(1)) if m else 0

    files.sort(key=extract_epoch, reverse=True)
    return files[0]


def find_gpt_model(speaker_folder):
    """GPT_weights_v2ProPlus 우선, 없으면 GPT_weights_v2Pro에서 탐색."""
    model = find_best_model(GPT_WEIGHT_DIR_PLUS, speaker_folder, ".ckpt")
    if model:
        return model
    return find_best_model(GPT_WEIGHT_DIR_PRO, speaker_folder, ".ckpt")


def find_sovits_model(speaker_folder):
    """SoVITS_weights_v2ProPlus에서 탐색."""
    return find_best_model(SOVITS_WEIGHT_DIR, speaker_folder, ".pth")


def get_texts_from_list(list_path, max_count):
    """
    .list 파일에서 대본 텍스트 추출.
    반환: [(audio_path, text), ...]
    """
    results = []
    try:
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                if len(results) >= max_count:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 4:
                    text = parts[3].strip()
                    if text:
                        results.append(text)
    except Exception as e:
        log(f"  .list 읽기 실패: {e}", "ERROR")
    return results


def set_model(gpt_path, sovits_path):
    """API 서버에 모델 교체 요청."""
    res_gpt    = requests.get(f"{API_BASE}/set_gpt_weights",
                              params={"weights_path": gpt_path}, timeout=30)
    res_sovits = requests.get(f"{API_BASE}/set_sovits_weights",
                              params={"weights_path": sovits_path}, timeout=30)
    if res_gpt.status_code != 200 or res_sovits.status_code != 200:
        raise RuntimeError(
            f"모델 교체 실패 | GPT({res_gpt.status_code}): {res_gpt.text} "
            f"| SoVITS({res_sovits.status_code}): {res_sovits.text}"
        )


def already_generated(speaker_folder):
    """이미 가짜 음성이 생성된 화자면 True (출력 폴더에 wav 파일 존재)."""
    out_dir = os.path.join(FAKE_VOICE_DIR, speaker_folder)
    if not os.path.isdir(out_dir):
        return False
    wav_files = glob.glob(os.path.join(out_dir, "*.wav"))
    return len(wav_files) > 0

#  메인 파이프라인
def main():
    os.makedirs(FAKE_VOICE_DIR, exist_ok=True)

    # 원천 음성 폴더 기준으로 화자 목록 구성
    speaker_folders = sorted([
        d for d in os.listdir(ORIGIN_VOICE_DIR)
        if os.path.isdir(os.path.join(ORIGIN_VOICE_DIR, d))
    ])

    log(f"총 {len(speaker_folders)}명 화자 발견")

    try:
        requests.get(f"{API_BASE}/", timeout=5)
        log("API 서버 연결 확인", "OK")
    except requests.exceptions.ConnectionError:
        log("API 서버에 연결할 수 없음. api_v2.bat 실행 확인할 것.", "ERROR")
        return
    except Exception:
        log("API 서버 연결 확인 (응답 있음)", "OK")

    print("=" * 60)

    success_list = []
    skip_list    = []
    fail_list    = []

    for idx, speaker_folder in enumerate(speaker_folders, 1):
        print()
        log(f"[{idx}/{len(speaker_folders)}] ▶ 화자: {speaker_folder}")

        # 이미 생성 완료된 화자 스킵
        if already_generated(speaker_folder):
            log(f"  이미 생성된 음성 존재 → 스킵", "WARN")
            skip_list.append(speaker_folder)
            continue

        # 모델 파일 탐색
        gpt_model    = find_gpt_model(speaker_folder)
        sovits_model = find_sovits_model(speaker_folder)

        if not gpt_model:
            log(f"  GPT 모델 없음 → 스킵 (학습 미완료)", "WARN")
            skip_list.append(speaker_folder)
            continue
        if not sovits_model:
            log(f"  SoVITS 모델 없음 → 스킵 (학습 미완료)", "WARN")
            skip_list.append(speaker_folder)
            continue

        log(f"  GPT    : {os.path.basename(gpt_model)}")
        log(f"  SoVITS : {os.path.basename(sovits_model)}")

        # ref_audio / prompt_text 탐색
        wav_dir  = os.path.join(ORIGIN_VOICE_DIR, speaker_folder)
        json_dir = os.path.join(ORIGIN_LABEL_DIR, speaker_folder)

        ref_audio = find_ref_audio(wav_dir)
        if not ref_audio:
            log(f"  ref_audio 없음 → 스킵", "ERROR")
            fail_list.append((speaker_folder, "ref_audio 없음"))
            continue

        prompt_text = get_prompt_text(ref_audio, json_dir)
        if not prompt_text:
            log(f"  prompt_text 추출 실패 → 스킵", "ERROR")
            fail_list.append((speaker_folder, "prompt_text 추출 실패"))
            continue

        dur = get_wav_duration(ref_audio)
        log(f"  ref_audio   : {os.path.basename(ref_audio)} ({dur:.1f}초)")
        log(f"  prompt_text : {prompt_text[:40]}{'...' if len(prompt_text) > 40 else ''}")

        # .list에서 대본 추출
        list_path = os.path.join(LIST_DIR, f"{speaker_folder}.list")
        if not os.path.exists(list_path):
            log(f"  .list 파일 없음: {list_path}", "ERROR")
            fail_list.append((speaker_folder, ".list 파일 없음"))
            continue

        texts = get_texts_from_list(list_path, MAX_COUNT)
        if not texts:
            log(f"  대본 추출 실패 → 스킵", "ERROR")
            fail_list.append((speaker_folder, "대본 추출 실패"))
            continue

        log(f"  대본 {len(texts)}개 확보")

        try:
            set_model(os.path.abspath(gpt_model), os.path.abspath(sovits_model))
            log(f"  모델 교체 완료", "OK")
        except Exception as e:
            log(f"  모델 교체 실패: {e}", "ERROR")
            fail_list.append((speaker_folder, f"모델 교체 실패: {e}"))
            continue

        time.sleep(1)  # 모델 로딩 대기

        out_dir = os.path.join(FAKE_VOICE_DIR, speaker_folder)
        os.makedirs(out_dir, exist_ok=True)

        generated = 0
        errors    = 0
        start_time = time.time()

        for i, text in enumerate(tqdm(texts, desc=f"  [{speaker_folder}] 생성 중", unit="개")):
            payload = {
                "ref_audio_path":    os.path.abspath(ref_audio),
                "prompt_text":       prompt_text,
                "prompt_lang":       PROMPT_LANG,
                "text":              text,
                "text_lang":         TEXT_LANG,
            }

            try:
                response = requests.post(
                    f"{API_BASE}/tts",
                    json=payload,
                    timeout=REQUEST_TIMEOUT
                )
                if response.status_code == 200:
                    out_path = os.path.join(out_dir, f"{speaker_folder}_FAKE_{i+1:06d}.wav")
                    with open(out_path, "wb") as f:
                        f.write(response.content)
                    generated += 1
                else:
                    log(f"\n  [{i+1}번] API 에러: {response.status_code}", "WARN")
                    errors += 1

            except Exception as e:
                log(f"\n  [{i+1}번] 통신 에러: {e}", "WARN")
                errors += 1

        elapsed = time.time() - start_time
        log(f"  ✔ 생성 완료: {generated}개 성공 / {errors}개 실패 / {elapsed:.1f}초", "OK")
        success_list.append(speaker_folder)

    print()
    print("=" * 60)
    log(f"파이프라인 완료")
    log(f"성공: {len(success_list)}명", "OK")
    log(f"스킵: {len(skip_list)}명 (이미 완료 or 모델 미완성)", "WARN")
    if fail_list:
        log(f"실패: {len(fail_list)}명", "ERROR")
        for name, reason in fail_list:
            log(f"  - {name}: {reason}", "ERROR")
    print("=" * 60)


if __name__ == "__main__":
    main()