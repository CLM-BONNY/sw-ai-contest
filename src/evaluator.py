import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


# 공통 지표 계산 함수 (내부용)
def _calculate_metrics(y_true, y_pred, y_prob=None):
    metrics = {
        "f1": f1_score(y_true, y_pred, average="weighted"),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


# 일반 모델용 (XGBoost 등)
def get_classification_metrics(y_true, y_pred, y_prob=None):
    return _calculate_metrics(y_true, y_pred, y_prob)


# HuggingFace Trainer 전용
def compute_metrics(pred):
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)

    # 이진 분류라면 확률값 추출
    y_prob = None
    try:
        if pred.predictions.shape[1] == 2:
            y_prob = pred.predictions[:, 1]
    except Exception:
        pass

    return _calculate_metrics(labels, preds, y_prob)
