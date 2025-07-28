from xgboost import XGBClassifier


def build_model(scale_pos_weight, n_estimators, random_state):
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        random_state=random_state,
    )
