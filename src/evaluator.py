import numpy as np
from sklearn.metrics import classification_report, f1_score, roc_auc_score


# ML 모델(XGBoost) 평가 지표 계산 함수
def print_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    print(report)


# Hugging Face Trainer 평가 지표 계산 함수
def compute_metrics(pred):
    # 정답 라벨 및 예측 결과 추출
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    f1 = f1_score(labels, preds, average="weighted")

    try:
        # ROC-AUC는 이진 분류에서 클래스 1의 확률값 필요
        auc = roc_auc_score(labels, pred.predictions[:, 1])
    except ValueError:
        # 예외 상황 처리 (예: 클래스가 하나뿐일 때)
        auc = 0.0

    return {
        "f1": f1,
        "roc_auc": auc,
    }
