from sklearn.metrics import roc_auc_score


def train_model(model, X_train_vec, y_train):
    model.fit(X_train_vec, y_train)
    return model


def evaluate_model(model, X_val_vec, y_val):
    val_probs = model.predict_proba(X_val_vec)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation AUC: {auc:.4f}")
    return auc


def predict(model, X_test_vec):
    return model.predict_proba(X_test_vec)[:, 1]
