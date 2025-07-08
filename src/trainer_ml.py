from sklearn.metrics import roc_auc_score
from src.evaluator import get_classification_metrics


def train_model(model, X_train_vec, y_train):
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_val_vec, y_val):
    val_probs = model.predict_proba(X_val_vec)[:, 1]
    val_preds = (val_probs > 0.5).astype(int)

    metrics = get_classification_metrics(y_val, val_preds, val_probs)

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics


def predict(model, X_test_vec):
    return model.predict_proba(X_test_vec)[:, 1]
