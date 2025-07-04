#!/bin/bash

echo "실험 실행 시작: Hugging Face 모델 학습"

# Step 1: 시드 고정, 로깅 등 초기 설정 확인
export TOKENIZERS_PARALLELISM=false

# Step 2: Hugging Face 모델 학습
export PYTHONPATH=$(pwd)
python experiments/train_hf.py

# Step 3: 학습 완료 모델 기반 추론 진행
export PYTHONPATH=$(pwd)
python experiments/infer_hf.py

echo "Hugging Face 학습 및 추론 완료"
