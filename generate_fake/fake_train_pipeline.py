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

GPTSOVITS_DIR = r"E:\GPT-SoVITS-v3lora-20250228"
PYTHON_EXEC = r"E:\GPT-SoVITS-v3lora-20250228\runtime\python.exe"
BASE_DATASET_DIR = r"E:\Project\AiyaAniya\datasets"
ORIGIN_VOICE_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice")
ORIGIN_LABEL_DIR = os.path.join(BASE_DATASET_DIR, "origin_voice_labeling")
LIST_OUTPUT_DIR = os.path.join(BASE_DATASET_DIR, "dataset_list")
EXP_ROOT = os.path.join(GPTSOVITS_DIR, "logs")
TEMP_DIR = os.path.join(GPTSOVITS_DIR, "TEMP")
VERSION = "v2ProPlus"
GPU_NUMBERS = "0"

PRETRAINED_S2G = "GPT_SoVITS/pretrained_models/v2pro/s2Gv2ProPlus.pth"
PRETRAINED_S2D = "GPT_SoVITS/pretrained_models/v2pro/s2Dv2ProPlus.pth"
PRETRAINED_S1  = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"

SSL_PRETRAINED_DIR  = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
BERT_PRETRAINED_DIR = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"

SOVITS_WEIGHT_DIR   = os.path.join(GPTSOVITS_DIR, "SoVITS_weights_v2ProPlus")
GPT_WEIGHT_DIR_PLUS = os.path.join(GPTSOVITS_DIR, "GPT_weights_v2ProPlus")
GPT_WEIGHT_DIR_PRO  = os.path.join(GPTSOVITS_DIR, "GPT_weights_v2Pro")

SOVITS_BATCH_SIZE            = 8
SOVITS_TOTAL_EPOCH           = 8
SOVITS_SAVE_EVERY_EPOCH      = 4
SOVITS_TEXT_LOW_LR_RATE      = 0.4
SOVITS_IF_SAVE_LATEST        = True
SOVITS_IF_SAVE_EVERY_WEIGHTS = True
SOVITS_IF_GRAD_CKPT          = False
SOVITS_LORA_RANK             = 128

GPT_BATCH_SIZE            = 8
GPT_TOTAL_EPOCH           = 15
GPT_SAVE_EVERY_EPOCH      = 5
GPT_IF_DPO                = False
GPT_IF_SAVE_LATEST        = True
GPT_IF_SAVE_EVERY_WEIGHTS = True

LANGUAGE = "ko"

MIN_REF_DURATION = 3.0

SKIP_SPEAKERS = [
]

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
        candidates.sort(key=lambda x: x[1]) 
        return candidates[0][1]

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
    log(f"    CMD: {cmd}")

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

    if new_files:
        log(f"    {step_name}: returncode={p.returncode} 이지만 "
            f"모델 파일 생성 확인됨 → 성공 처리", "WARN")
        log(f"    생성된 파일: {sorted(new_files)}", "OK")
        return

    raise RuntimeError(f"{step_name} 실패 (returncode={p.returncode}, 생성된 모델 없음)")


def step_1a_get_text(exp_name, list_path, wav_dir, gpu_index):
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

    txt_part  = os.path.join(opt_dir, "2-name2text-0.txt")
    txt_final = os.path.join(opt_dir, "2-name2text.txt")
    if os.path.exists(txt_part):
        shutil.move(txt_part, txt_final)


def step_1b_get_hubert(exp_name, list_path, wav_dir, gpu_index):
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

    if "Pro" in VERSION:
        run_step("1b-get-sv",
                 f'"{PYTHON_EXEC}" -s GPT_SoVITS/prepare_datasets/2-get-sv.py', env)


def step_1c_get_semantic(exp_name, list_path, gpu_index):
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

    tsv_part  = os.path.join(opt_dir, "6-name2semantic-0.tsv")
    tsv_final = os.path.join(opt_dir, "6-name2semantic.tsv")
    if os.path.exists(tsv_part) and not os.path.exists(tsv_final):
        shutil.move(tsv_part, tsv_final)
        log("    6-name2semantic-0.tsv → 6-name2semantic.tsv 병합 완료", "OK")


def step_train_sovits(exp_name):
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

    run_step_check_output(
        "SoVITS 학습",
        f'"{PYTHON_EXEC}" -s GPT_SoVITS/s2_train.py --config "{tmp_config_path}"',
        env,
        check_dir=SOVITS_WEIGHT_DIR,
        check_prefix=exp_name,
    )


def step_train_gpt(exp_name):
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
    sovits_files = glob.glob(os.path.join(SOVITS_WEIGHT_DIR, f"{exp_name}*.pth"))
    return len(sovits_files) > 0

def is_gpt_trained(exp_name):
    gpt_files = glob.glob(os.path.join(GPT_WEIGHT_DIR_PLUS, f"{exp_name}*.ckpt")) + \
                glob.glob(os.path.join(GPT_WEIGHT_DIR_PRO,  f"{exp_name}*.ckpt"))
    return len(gpt_files) > 0

def is_preprocess_done(exp_name):
    opt_dir = os.path.join(EXP_ROOT, exp_name)
    text_done    = os.path.exists(os.path.join(opt_dir, "2-name2text.txt"))
    semantic_done = os.path.exists(os.path.join(opt_dir, "6-name2semantic.tsv"))
    return text_done, semantic_done

def is_already_trained(exp_name):
    return is_sovits_trained(exp_name) and is_gpt_trained(exp_name)


def main():
    os.chdir(GPTSOVITS_DIR)

    os.makedirs(LIST_OUTPUT_DIR, exist_ok=True)
    os.makedirs(EXP_ROOT, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(SOVITS_WEIGHT_DIR, exist_ok=True)
    os.makedirs(GPT_WEIGHT_DIR_PLUS, exist_ok=True)
    os.makedirs(GPT_WEIGHT_DIR_PRO, exist_ok=True)

    gpu_index = int(GPU_NUMBERS.split("-")[0])

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

        if speaker_folder in SKIP_SPEAKERS:
            log(f"  SKIP_SPEAKERS 목록에 있어 스킵", "WARN")
            continue

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

        text_done, semantic_done = is_preprocess_done(speaker_folder)
        sovits_done = is_sovits_trained(speaker_folder)
        gpt_done    = is_gpt_trained(speaker_folder)

        try:
            log("  [STEP 0] .list 파일 생성")
            if os.path.exists(list_path):
                log(f"  .list 이미 존재 → 재사용", "WARN")
            else:
                count = generate_list_file(speaker_folder, wav_dir, json_dir, list_path)
                if count == 0:
                    raise RuntimeError(".list 생성 실패 (0개 항목)")
                log(f"  .list 생성 완료 ({count}개)", "OK")

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
            continue

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
