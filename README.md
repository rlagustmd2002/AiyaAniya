# 아이(AI)야 아니야 — AI 합성 음성 탐지 시스템

본 프로젝트는 실제 사람의 육성과 AI가 합성한 가짜 음성(딥보이스)을 구별하는 시스템입니다.
Wav2Vec2를 fine-tuning 하여 입력된 음성이 진짜인지 합성인지를 판별하고, 이를 웹 앱 형태로 제공합니다.

- **모델**: `facebook/wav2vec2-base` 기반 fine-tuning + MLP 분류기
- **성능**: 학습에 사용하지 않은 처음 보는 화자 기준 정확도 **97.78%**, AUC **0.9888**
- **데이터**: AI Hub 다화자 음성합성 데이터(진짜) + GPT-SoVITS로 합성한 가짜 음성

## 구성

- `train.py`, `test.py` : 모델 학습 및 평가
- `generate_fake/` : GPT-SoVITS를 이용한 가짜 음성 생성 파이프라인 (GPT-SoVITS 설치 폴더 안에 두고 실행)
- `backend/` : FastAPI 기반 추론 서버 및 웹 UI (`main.py`, `static/index.html`)
- `results/` : 학습 곡선, 평가 지표 등 결과물

## 실행 방법

```bash
pip install -r requirements.txt

python train.py          # 학습
python test.py           # 평가

cd backend && python main.py   # 웹 앱 (브라우저에서 http://localhost:8000)
```

## 참고

- **학습된 모델(`best_model.pth`)과 GPT-SoVITS 음성 합성 모델은 파일 용량이 커서 본 저장소에 포함하지 않았습니다.** 모델은 위 코드로 직접 학습하여 생성하면 됩니다.
- 데이터셋(AI Hub 다화자 음성합성 데이터) 역시 용량 및 이용 약관 문제로 포함하지 않았으며, AI Hub에서 직접 받아 사용해야 합니다.
- 가짜 음성 생성을 위해서는 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS)를 별도로 설치하고, `generate_fake/`의 스크립트를 그 폴더 안에서 실행해야 합니다.

## 출처

- Wav2Vec2 (Baevski et al., 2020) · `facebook/wav2vec2-base` (Hugging Face)
- GPT-SoVITS (RVC-Boss)
- AI Hub 다화자 음성합성 데이터 (한국지능정보사회진흥원, NIA)
