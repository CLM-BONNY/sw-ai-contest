import pandas as pd

# 예측 결과 파일 경로 리스트
files = [
    "submission_koelectra_v3_discriminator.csv",
    "submission_KcELECTRA_2022.csv",
    "submission_klue_roberta.csv",
    "submission_daekeun-ml_koelectra-small-v3-nsmc.csv",
]

# 각 모델의 가중치 (총합 = 1)
weights = [0.35, 0.30, 0.20, 0.15]

# 각 파일 읽기
dfs = [pd.read_csv(file) for file in files]

# weighted soft voting 수행
weighted_probs = sum(w * df["generated"] for w, df in zip(weights, dfs))

# 결과 저장
final_submission = dfs[0][["ID"]].copy()
final_submission["generated"] = weighted_probs

# 파일 저장
final_submission.to_csv("submission_weighted_soft_voting_2.csv", index=False)
print("✅ weighted_soft_voting_submission.csv 저장 완료!")
