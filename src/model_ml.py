from xgboost import XGBClassifier


def build_model():
    return XGBClassifier(random_state=42)
