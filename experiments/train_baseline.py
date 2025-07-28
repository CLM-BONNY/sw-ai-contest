import yaml
import time
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

from src.vectorizer import get_vectorizer
from src.model_ml import build_model
from src.trainer_ml import train_model, evaluate_model, predict
from utils.logger import get_logger, log_metrics

# config 파일 내부 설정 로딩
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

project_name = config["project_name"]
entity = "clm-bonny-203ho"
model_name = config["model_name"]
train_path = config["train_path"]
predict_path = config["predict_path"]
test_path = config["test_path"]
vectorizer_type = config["vectorizer"]
random_state = config["seed"]
n_estimators = config["n_estimators"]

# scale_pos_weights 값
scale_pos_weights = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15]

for scale_pos_weight in scale_pos_weights:
    # run name 설정
    run_name = f"{model_name}-w{scale_pos_weight}-n{n_estimators}"

    # 로거 설정
    logger = get_logger("XGBoost-Baseline")

    # W&B 설정
    wandb.init(
        project=project_name,
        entity=entity,
        name=run_name,
        config={
            "model": model_name,
            "vectorizer": vectorizer_type,
            "scale_pos_weight": scale_pos_weight,
            "random_state": random_state,
            "n_estimators": n_estimators,
        },
    )

    # 데이터 로딩
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")

    # 라벨과 입력 분리
    X = train[["title", "full_text"]]
    y = train["generated"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 벡터화
    print("벡터화 시작")
    start_time = time.time()
    vectorizer = get_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    print(f"소요 시간: {time.time() - start_time:.2f}초")

    # 모델 학습 및 평가
    print("모델 학습 및 평가 시작")
    model = build_model(scale_pos_weight, n_estimators, random_state)
    model = train_model(model, X_train_vec, y_train)
    metrics = evaluate_model(model, X_val_vec, y_val)

    # 콘솔 + 파일 + W&B에 기록
    log_metrics(logger, metrics)

    wandb.finish()
    print("모델 학습 및 평가 완료")

    # 테스트셋 전처리 및 예측
    print("테스트셋 전처리 및 예측 시작")
    test = test.rename(columns={"paragraph_text": "full_text"})
    X_test = test[["title", "full_text"]]
    X_test_vec = vectorizer.transform(X_test)
    test_probs = predict(model, X_test_vec)
    print("테스트셋 전처리 및 예측 완료")

    # 결과 저장
    print("결과 저장 시작")
    sample_submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    sample_submission["generated"] = test_probs
    predict_path = f"data/raw/baseline_submission_{str(scale_pos_weight)}.csv"
    sample_submission.to_csv(predict_path, index=False)
    print("결과 저장 완료")
