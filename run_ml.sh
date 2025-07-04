#!/bin/bash

echo "베이스라인 실험 시작: TF-IDF + ML 모델"

# 환경 설정
export PYTHONPATH=$(pwd)

# 실행
python experiments/train_baseline.py

echo "베이스라인 실험 완료"
