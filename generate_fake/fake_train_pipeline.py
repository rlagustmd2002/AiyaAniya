"""
GPT-SoVITS v2ProPlus 자동 학습 파이프라인
=========================================
40명 화자의 음성/라벨 데이터를 순차적으로 읽어
각 화자별 .list 생성 → 전처리(1a/1b/1c) → SoVITS 학습 → GPT 학습을
자동으로 실행합니다.

★ 실행 방법:
  1. 이 파일을 GPT-SoVITS 설치 폴더에 복사하세요.
     예) E:\\GPT-SoVITS-v3lora-20250228\\auto_train_pipeline.py

  2. 아래 [설정] 섹션의 경로들을 확인하세요.

  3. cmd에서 GPT-SoVITS 폴더로 이동 후 실행:
     cd E:\\GPT-SoVITS-v3lora-20250228
     runtime\\python.exe auto_train_pipeline.py

  ※ 반드시 GPT-SoVITS 설치 폴더에서 실행해야 합니다.
     (내부 스크립트들이 상대 경로를 사용하기 때문)
"""

import os
import sys
import json
import time
import wave
import glob
import shutil
import subprocess
import yaml
import traceback
from pathlib import Path
from subprocess import Popen
from tqdm import tqdm

# ============================================================
#  ★ [설정] - 여기만 수정하세요
# ============================================================

# GPT-SoVITS 설치 폴더 (이 스크립트를 여기서 실행하세요)
GPTSOVITS_DIR = r"E:\GPT-SoVITS-v3lora-20250228"

# Python 실행 경로
PYTHON_EXEC = r"E:\GPT-SoVITS-v3lora-20250228\runtime\python.exe"

# 데이터셋 루트
BASE_DATASET_DIR = r"E:\Project\AiyaAniya\datasets"

# 원천 음성 폴더 (WAV)
ORIGIN_VOICE_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice")

# 라벨링 폴더 (JSON)
ORIGIN_LABEL_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice_labeling")

# .list 파일 출력 폴더
LIST_OUTPUT_DIR = os.path.join(BASE_DATASET_DIR, "dataset_list")

# 학습 실험 결과 저장 폴더 (webui의 exp_root에 해당)
EXP_ROOT = os.path.join(GPTSOVITS_DIR, "logs")

# TEMP 폴더
TEMP_DIR = os.path.join(GPTSOVITS_DIR, "TEMP")

# 학습 버전
VERSION = "v2ProPlus"

# GPU 번호 (단일 GPU면 "0", 멀티면 "0-1")
GPU_NUMBERS = "0"

# Pretrained 모델 경로 (v2ProPlus 기준)
PRETRAINED_S2G = "GPT_SoVITS/pretrained_models/v2pro/s2Gv2ProPlus.pth"
PRETRAINED_S2D = "GPT_SoVITS/pretrained_models/v2pro/s2Dv2ProPlus.pth"
PRETRAINED_S1  = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

# SSL / BERT pretrained 경로
SSL_PRETRAINED_DIR  = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_PRETRAINED_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

# SoVITS / GPT 가중치 저장 폴더
SOVITS_WEIGHT_DIR   = os.path.join(GPTSOVITS_DIR, "SoVITS_weights_v2ProPlus")
GPT_WEIGHT_DIR_PLUS = os.path.join(GPTSOVITS_DIR, "GPT_weights_v2ProPlus")
GPT_WEIGHT_DIR_PRO  = os.path.join(GPTSOVITS_DIR, "GPT_weights_v2Pro")

# SoVITS 학습 파라미터
SOVITS_BATCH_SIZE            = 8
SOVITS_TOTAL_EPOCH           = 8
SOVITS_SAVE_EVERY_EPOCH      = 4
SOVITS_TEXT_LOW_LR_RATE      = 0.4
SOVITS_IF_SAVE_LATEST        = True
SOVITS_IF_SAVE_EVERY_WEIGHTS = True
SOVITS_IF_GRAD_CKPT          = False
SOVITS_LORA_RANK             = 128

# GPT 학습 파라미터
GPT_BATCH_SIZE            = 8
GPT_TOTAL_EPOCH           = 15
GPT_SAVE_EVERY_EPOCH      = 5
GPT_IF_DPO                = False
GPT_IF_SAVE_LATEST        = True
GPT_IF_SAVE_EVERY_WEIGHTS = True

# 언어
LANGUAGE = "ko"

# ref_audio 최소 길이 (초) - 이 이상인 파일 우선 선택
MIN_REF_DURATION = 3.0

# ★ 이미 학습 완료된 화자 폴더명 목록 (스킵됩니다)
# 예: SKIP_SPEAKERS = ["0005_G1A3E7_KYG", "0007_G1A2E7_KES"]
SKIP_SPEAKERS = [
]

# ============================================================
#  내부 유틸 함수
# ============================================================

def log(msg, level="INFO"):
    tag = {"INFO": "[ INFO ]", "OK": "[  OK  ]", "WARN": "[ WARN ]", "ERROR": "[ERROR ]"}.get(level, "[ INFO ]")
    print(f"{tag} {msg}", flush=True)


def get_wav_duration(wav_path):
    """WAV 재생 시간(초) 반환. 실패 시 0.0"""
    try:
        with wave.open(wav_path, "r") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def find_ref_audio(wav_dir):
    """
    화자 폴더에서 MIN_REF_DURATION 이상인 WAV를 파일명 순으로 정렬해 첫 번째 반환.
    없으면 가장 긴 파일 반환.
    """
    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_files:
        return None

    candidates = [(get_wav_duration(w), w) for w in wav_files if get_wav_duration(w) >= MIN_REF_DURATION]
    if candidates:
        candidates.sort(key=lambda x: x[1])  # 파일명 순
        return candidates[0][1]

    # 3초 이상 없으면 가장 긴 파일
    all_durs = [(get_wav_duration(w), w) for w in wav_files]
    all_durs.sort(reverse=True)
    return all_durs[0][1]


def get_prompt_text(wav_path, json_dir):
    """ref_audio에 대응하는 JSON에서 TransLabelText 추출."""
    json_name = os.path.splitext(os.path.basename(wav_path))[0] + ".json"
    json_path = os.path.join(json_dir, json_name)
    if not os.path.exists(json_path):
        log(f"JSON 없음: {json_path}", "WARN")
        return ""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)["전사정보"]["TransLabelText"].strip()
    except Exception as e:
        log(f"JSON 읽기 실패: {e}", "WARN")
        return ""


def generate_list_file(speaker_folder, wav_dir, json_dir, list_path):
    """JSON 라벨을 읽어 .list 파일 생성. 생성된 라인 수 반환."""
    json_files = sorted([f for f in os.listdir(json_dir) if f.endswith(".json")])
    if not json_files:
        return 0

    speaker = None
    count = 0
    with open(list_path, "w", encoding="utf-8") as out_f:
        for filename in tqdm(json_files, desc=f"  list생성", unit="file", leave=False):
            try:
                with open(os.path.join(json_dir, filename), "r", encoding="utf-8") as jf:
                    data = json.load(jf)
                wav_filename = data["파일정보"]["FileName"]
                text = data["전사정보"]["TransLabelText"].strip()
                if speaker is None:
                    speaker = data["화자정보"]["SpeakerName"]
                if not text:
                    continue
                audio_path = os.path.join(wav_dir, wav_filename)
                out_f.write(f"{audio_path}|{speaker}|{LANGUAGE}|{text}\n")
                count += 1
            except Exception as e:
                log(f"    [SKIP] {filename} → {e}", "WARN")
    return count


def run_step(step_name, cmd, env):
    """subprocess 실행 후 완료 대기. 실패 시 RuntimeError."""
    log(f"    CMD: {cmd}")
    p = Popen(cmd, shell=True, env=env)
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"{step_name} 실패 (returncode={p.returncode})")
    log(f"    완료: {step_name}", "OK")


def run_step_check_output(step_name, cmd, env, check_dir, check_prefix):
    """
    subprocess 실행 후, returncode가 0이 아니어도
    결과 모델 파일이 생성됐으면 성공으로 간주.

    Windows + PyTorch 멀티프로세싱 환경에서 학습은 정상 완료됐는데
    프로세스 종료 단계에서 PermissionError(WinError 5)가 발생해
    returncode=1이 반환되는 버그를 우회하기 위함.
    """
    log(f"    CMD: {cmd}")

    # 실행 전 기존 파일 목록 기록
    def list_models():
        if not os.path.isdir(check_dir):
            return set()
        return {
            f for f in os.listdir(check_dir)
            if f.startswith(check_prefix) and f.endswith(".pth")
        }

    before = list_models()

    p = Popen(cmd, shell=True, env=env)
    p.wait()

    after = list_models()
    new_files = after - before

    if p.returncode == 0:
        log(f"    완료: {step_name}", "OK")
        return

    # returncode != 0 이지만 새 모델 파일이 생겼으면 성공으로 간주
    if new_files:
        log(f"    {step_name}: returncode={p.returncode} 이지만 "
            f"모델 파일 생성 확인됨 → 성공 처리", "WARN")
        log(f"    생성된 파일: {sorted(new_files)}", "OK")
        return

    # 새 파일도 없으면 진짜 실패
    raise RuntimeError(f"{step_name} 실패 (returncode={p.returncode}, 생성된 모델 없음)")


# ============================================================
#  학습 단계 함수
# ============================================================

def step_1a_get_text(exp_name, list_path, wav_dir, gpu_index):
    """1a: 텍스트 전처리 (1-get-text.py)"""
    opt_dir = os.path.join(EXP_ROOT, exp_name)
    os.makedirs(opt_dir, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "inp_text":              list_path,
        "inp_wav_dir":           wav_dir,
        "exp_name":              exp_name,
        "opt_dir":               opt_dir,
        "bert_pretrained_dir":   BERT_PRETRAINED_DIR,
        "i_part":                "0",
        "all_parts":             "1",
        "_CUDA_VISIBLE_DEVICES": str(gpu_index),
        "is_half":               "True",
    })

    cmd = f'"{PYTHON_EXEC}" -s GPT_SoVITS/prepare_datasets/1-get-text.py'
    run_step("1a-get-text", cmd, env)

    # 파트 파일 병합 (webui와 동일)
    txt_part  = os.path.join(opt_dir, "2-name2text-0.txt")
    txt_final = os.path.join(opt_dir, "2-name2text.txt")
    if os.path.exists(txt_part):
        shutil.move(txt_part, txt_final)


def step_1b_get_hubert(exp_name, list_path, wav_dir, gpu_index):
    """1b: SSL 특징 추출 (2-get-hubert-wav32k.py + 2-get-sv.py)"""
    opt_dir = os.path.join(EXP_ROOT, exp_name)
    sv_path = "GPT_SoVITS/pretrained_models/sv/pretrained_eres2netv2w24s4ep4.ckpt"

    env = os.environ.copy()
    env.update({
        "inp_text":              list_path,
        "inp_wav_dir":           wav_dir,
        "exp_name":              exp_name,
        "opt_dir":               opt_dir,
        "cnhubert_base_dir":     SSL_PRETRAINED_DIR,
        "sv_path":               sv_path,
        "i_part":                "0",
        "all_parts":             "1",
        "_CUDA_VISIBLE_DEVICES": str(gpu_index),
        "is_half":               "True",
    })

    run_step("1b-get-hubert-wav32k",
             f'"{PYTHON_EXEC}" -s GPT_SoVITS/prepare_datasets/2-get-hubert-wav32k.py', env)

    # v2Pro / v2ProPlus 는 추가로 sv 추출
    if "Pro" in VERSION:
        run_step("1b-get-sv",
                 f'"{PYTHON_EXEC}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py', env)


def step_1c_get_semantic(exp_name, list_path, gpu_index):
    """1c: 의미 토큰 추출 (3-get-semantic.py)"""
    opt_dir = os.path.join(EXP_ROOT, exp_name)
    config_file = (
        f"GPT_SoVITS/configs/s2{VERSION}.json"
        if VERSION in {"v2Pro", "v2ProPlus"}
        else "GPT_SoVITS/configs/s2.json"
    )

    env = os.environ.copy()
    env.update({
        "inp_text":              list_path,
        "exp_name":              exp_name,
        "opt_dir":               opt_dir,
        "pretrained_s2G":        PRETRAINED_S2G,
        "s2config_path":         config_file,
        "i_part":                "0",
        "all_parts":             "1",
        "_CUDA_VISIBLE_DEVICES": str(gpu_index),
        "is_half":               "True",
    })

    run_step("1c-get-semantic",
             f'"{PYTHON_EXEC}" -s GPT_SoVITS/prepare_datasets/3-get-semantic.py', env)

    # 파트 파일 병합 (6-name2semantic-0.tsv → 6-name2semantic.tsv)
    tsv_part  = os.path.join(opt_dir, "6-name2semantic-0.tsv")
    tsv_final = os.path.join(opt_dir, "6-name2semantic.tsv")
    if os.path.exists(tsv_part) and not os.path.exists(tsv_final):
        shutil.move(tsv_part, tsv_final)
        log("    6-name2semantic-0.tsv → 6-name2semantic.tsv 병합 완료", "OK")


def step_train_sovits(exp_name):
    """SoVITS 파인튜닝 학습 (s2_train.py)"""
    s2_dir = os.path.join(EXP_ROOT, exp_name)
    os.makedirs(os.path.join(s2_dir, f"logs_s2_{VERSION}"), exist_ok=True)

    config_file = (
        f"GPT_SoVITS/configs/s2{VERSION}.json"
        if VERSION in {"v2Pro", "v2ProPlus"}
        else "GPT_SoVITS/configs/s2.json"
    )

    with open(config_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["train"]["batch_size"]            = SOVITS_BATCH_SIZE
    data["train"]["epochs"]                = SOVITS_TOTAL_EPOCH
    data["train"]["text_low_lr_rate"]      = SOVITS_TEXT_LOW_LR_RATE
    data["train"]["pretrained_s2G"]        = PRETRAINED_S2G
    data["train"]["pretrained_s2D"]        = PRETRAINED_S2D
    data["train"]["if_save_latest"]        = SOVITS_IF_SAVE_LATEST
    data["train"]["if_save_every_weights"] = SOVITS_IF_SAVE_EVERY_WEIGHTS
    data["train"]["save_every_epoch"]      = SOVITS_SAVE_EVERY_EPOCH
    data["train"]["gpu_numbers"]           = GPU_NUMBERS
    data["train"]["grad_ckpt"]             = SOVITS_IF_GRAD_CKPT
    data["train"]["lora_rank"]             = SOVITS_LORA_RANK
    data["model"]["version"]               = VERSION
    data["data"]["exp_dir"]                = s2_dir
    data["s2_ckpt_dir"]                    = s2_dir
    data["save_weight_dir"]                = SOVITS_WEIGHT_DIR
    data["name"]                           = exp_name
    data["version"]                        = VERSION

    tmp_config_path = os.path.join(TEMP_DIR, f"tmp_s2_{exp_name}.json")
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    env = os.environ.copy()
    env["_CUDA_VISIBLE_DEVICES"] = GPU_NUMBERS.split("-")[0]

    # SoVITS 모델은 SOVITS_WEIGHT_DIR에 "{exp_name}_e..." 형태로 저장됨
    # Windows 멀티프로세싱 종료 버그(WinError 5)로 returncode=1이 나와도
    # 모델 파일이 생성됐으면 성공으로 간주
    run_step_check_output(
        "SoVITS 학습",
        f'"{PYTHON_EXEC}" -s GPT_SoVITS/s2_train.py --config "{tmp_config_path}"',
        env,
        check_dir=SOVITS_WEIGHT_DIR,
        check_prefix=exp_name,
    )


def step_train_gpt(exp_name):
    """GPT 파인튜닝 학습 (s1_train.py)"""
    s1_dir = os.path.join(EXP_ROOT, exp_name)
    os.makedirs(os.path.join(s1_dir, "logs_s1"), exist_ok=True)

    with open("GPT_SoVITS/configs/s1longer-v2.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data["train"]["batch_size"]            = GPT_BATCH_SIZE
    data["train"]["epochs"]                = GPT_TOTAL_EPOCH
    data["pretrained_s1"]                  = PRETRAINED_S1
    data["train"]["save_every_n_epoch"]    = GPT_SAVE_EVERY_EPOCH
    data["train"]["if_save_every_weights"] = GPT_IF_SAVE_EVERY_WEIGHTS
    data["train"]["if_save_latest"]        = GPT_IF_SAVE_LATEST
    data["train"]["if_dpo"]                = GPT_IF_DPO
    # GPT weights가 v2Pro에 저장되는 경우가 있어 둘 다 생성
    data["train"]["half_weights_save_dir"] = GPT_WEIGHT_DIR_PLUS
    data["train"]["exp_name"]              = exp_name
    data["train_semantic_path"]            = os.path.join(s1_dir, "6-name2semantic.tsv")
    data["train_phoneme_path"]             = os.path.join(s1_dir, "2-name2text.txt")
    data["output_dir"]                     = os.path.join(s1_dir, f"logs_s1_{VERSION}")

    tmp_config_path = os.path.join(TEMP_DIR, f"tmp_s1_{exp_name}.yaml")
    with open(tmp_config_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

    env = os.environ.copy()
    env["_CUDA_VISIBLE_DEVICES"] = GPU_NUMBERS.replace("-", ",")
    env["hz"] = "25hz"

    # GPT 모델은 .ckpt로 GPT_WEIGHT_DIR_PLUS 또는 GPT_WEIGHT_DIR_PRO에 저장됨
    # SoVITS와 동일한 Windows 멀티프로세싱 종료 버그 대응

    def list_gpt_ckpts():
        files = set()
        for d in (GPT_WEIGHT_DIR_PLUS, GPT_WEIGHT_DIR_PRO):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.startswith(exp_name) and f.endswith(".ckpt"):
                        files.add(os.path.join(d, f))
        return files

    before = list_gpt_ckpts()

    cmd = f'"{PYTHON_EXEC}" -s GPT_SoVITS/s1_train.py --config_file "{tmp_config_path}"'
    log(f"    CMD: {cmd}")
    p = Popen(cmd, shell=True, env=env)
    p.wait()

    after = list_gpt_ckpts()
    new_files = after - before

    if p.returncode == 0:
        log("    완료: GPT 학습", "OK")
    elif new_files:
        log(f"    GPT 학습: returncode={p.returncode} 이지만 "
            f"모델 파일 생성 확인됨 → 성공 처리", "WARN")
        log(f"    생성된 파일: {sorted(os.path.basename(f) for f in new_files)}", "OK")
    else:
        raise RuntimeError(f"GPT 학습 실패 (returncode={p.returncode}, 생성된 모델 없음)")


def is_sovits_trained(exp_name):
    """SoVITS 모델 파일이 존재하면 True."""
    sovits_files = glob.glob(os.path.join(SOVITS_WEIGHT_DIR, f"{exp_name}*.pth"))
    return len(sovits_files) > 0

def is_gpt_trained(exp_name):
    """GPT 모델 파일이 존재하면 True."""
    gpt_files = glob.glob(os.path.join(GPT_WEIGHT_DIR_PLUS, f"{exp_name}*.ckpt")) + \
                glob.glob(os.path.join(GPT_WEIGHT_DIR_PRO,  f"{exp_name}*.ckpt"))
    return len(gpt_files) > 0

def is_preprocess_done(exp_name):
    """전처리 완료 여부 확인 (2-name2text.txt, 6-name2semantic.tsv 존재 여부)."""
    opt_dir = os.path.join(EXP_ROOT, exp_name)
    text_done    = os.path.exists(os.path.join(opt_dir, "2-name2text.txt"))
    semantic_done = os.path.exists(os.path.join(opt_dir, "6-name2semantic.tsv"))
    return text_done, semantic_done

def is_already_trained(exp_name):
    """SoVITS + GPT 모델 파일이 모두 존재하면 True (학습 완료 간주)."""
    return is_sovits_trained(exp_name) and is_gpt_trained(exp_name)


# ============================================================
#  메인 파이프라인
# ============================================================

def main():
    # GPT-SoVITS 폴더를 작업 디렉토리로 설정 (상대경로 스크립트 때문에 필수)
    os.chdir(GPTSOVITS_DIR)

    os.makedirs(LIST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXP_ROOT, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SOVITS_WEIGHT_DIR, exist_ok=True)
    os.makedirs(GPT_WEIGHT_DIR_PLUS, exist_ok=True)
    os.makedirs(GPT_WEIGHT_DIR_PRO, exist_ok=True)

    gpu_index = int(GPU_NUMBERS.split("-")[0])

    # 화자 폴더 목록
    speaker_folders = sorted([
        d for d in os.listdir(ORIGIN_VOICE_DIR)
        if os.path.isdir(os.path.join(ORIGIN_VOICE_DIR, d))
    ])

    log(f"총 {len(speaker_folders)}명 화자 발견")
    log(f"스킵 지정: {SKIP_SPEAKERS if SKIP_SPEAKERS else '없음'}")
    print("=" * 60)

    success_list = []
    fail_list    = []

    for idx, speaker_folder in enumerate(speaker_folders, 1):
        print()
        log(f"[{idx}/{len(speaker_folders)}] ▶ 화자: {speaker_folder}")

        # SKIP_SPEAKERS 목록
        if speaker_folder in SKIP_SPEAKERS:
            log(f"  SKIP_SPEAKERS 목록에 있어 스킵", "WARN")
            continue

        # SoVITS + GPT 둘 다 완료 → 전체 스킵
        if is_already_trained(speaker_folder):
            log(f"  SoVITS + GPT 모두 완료 → 스킵", "WARN")
            success_list.append(speaker_folder)
            continue

        wav_dir   = os.path.join(ORIGIN_VOICE_DIR, speaker_folder)
        json_dir  = os.path.join(ORIGIN_LABEL_DIR, speaker_folder)
        list_path = os.path.join(LIST_OUTPUT_DIR, f"{speaker_folder}.list")

        if not os.path.isdir(wav_dir):
            log(f"  WAV 폴더 없음: {wav_dir}", "ERROR")
            fail_list.append((speaker_folder, "WAV 폴더 없음"))
            continue
        if not os.path.isdir(json_dir):
            log(f"  JSON 폴더 없음: {json_dir}", "ERROR")
            fail_list.append((speaker_folder, "JSON 폴더 없음"))
            continue

        # 전처리 완료 여부 사전 확인
        text_done, semantic_done = is_preprocess_done(speaker_folder)
        sovits_done = is_sovits_trained(speaker_folder)
        gpt_done    = is_gpt_trained(speaker_folder)

        try:
            # ── STEP 0: .list 파일 생성 ──────────────────────────
            log("  [STEP 0] .list 파일 생성")
            if os.path.exists(list_path):
                log(f"  .list 이미 존재 → 재사용", "WARN")
            else:
                count = generate_list_file(speaker_folder, wav_dir, json_dir, list_path)
                if count == 0:
                    raise RuntimeError(".list 생성 실패 (0개 항목)")
                log(f"  .list 생성 완료 ({count}개)", "OK")

            # ref_audio / prompt_text 로그 출력
            ref_audio = find_ref_audio(wav_dir)
            if ref_audio:
                prompt_text = get_prompt_text(ref_audio, json_dir)
                dur = get_wav_duration(ref_audio)
                log(f"  ref_audio : {os.path.basename(ref_audio)} ({dur:.1f}초)")
                log(f"  prompt_text: {prompt_text[:40]}{'...' if len(prompt_text)>40 else ''}")

            # ── STEP 1a: 텍스트 전처리 ───────────────────────────
            if text_done:
                log("  [STEP 1a] 텍스트 전처리 → 이미 완료, 스킵", "WARN")
            else:
                log("  [STEP 1a] 텍스트 전처리")
                step_1a_get_text(speaker_folder, list_path, wav_dir, gpu_index)

            # ── STEP 1b: SSL 특징 추출 ───────────────────────────
            # hubert 결과물은 별도 파일로 확인 어려우므로 text_done 기준으로 같이 스킵
            if text_done and semantic_done:
                log("  [STEP 1b] SSL 특징 추출 → 이미 완료, 스킵", "WARN")
            else:
                log("  [STEP 1b] SSL 특징 추출")
                step_1b_get_hubert(speaker_folder, list_path, wav_dir, gpu_index)

            # ── STEP 1c: 의미 토큰 추출 ──────────────────────────
            if semantic_done:
                log("  [STEP 1c] 의미 토큰 추출 → 이미 완료, 스킵", "WARN")
            else:
                log("  [STEP 1c] 의미 토큰 추출")
                step_1c_get_semantic(speaker_folder, list_path, gpu_index)

            # ── STEP 2: SoVITS 학습 ──────────────────────────────
            if sovits_done:
                log("  [STEP 2] SoVITS 학습 → 이미 완료, 스킵", "WARN")
            else:
                log("  [STEP 2] SoVITS 학습")
                step_train_sovits(speaker_folder)

            # ── STEP 3: GPT 학습 ─────────────────────────────────
            if gpt_done:
                log("  [STEP 3] GPT 학습 → 이미 완료, 스킵", "WARN")
            else:
                log("  [STEP 3] GPT 학습")
                step_train_gpt(speaker_folder)

            log(f"  ✔ [{speaker_folder}] 완료!", "OK")
            success_list.append(speaker_folder)

        except Exception as e:
            log(f"  ✘ [{speaker_folder}] 실패: {e}", "ERROR")
            traceback.print_exc()
            fail_list.append((speaker_folder, str(e)))
            continue  # 실패해도 다음 화자로 진행

    # ── 최종 요약 ─────────────────────────────────────────────
    print()
    print("=" * 60)
    log(f"파이프라인 완료! 성공 {len(success_list)}명 / 실패 {len(fail_list)}명")
    if success_list:
        log(f"성공 목록: {success_list}", "OK")
    if fail_list:
        log("실패 목록:", "ERROR")
        for name, reason in fail_list:
            log(f"  - {name}: {reason}", "ERROR")
    print("=" * 60)


if __name__ == "__main__":
    main()