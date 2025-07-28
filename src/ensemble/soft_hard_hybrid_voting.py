import pandas as pd

# 예측 결과 파일 리스트
files = [
    "submission_koelectra_v3_discriminator.csv",
    "submission_KcELECTRA_2022.csv",
    "submission_klue_roberta.csv",
    "submission_daekeun-ml_koelectra-small-v3-nsmc.csv",
]

# 파일 불러오기
dfs = [pd.read_csv(file) for file in files]

# 각 모델의 예측 확률을 0.5 기준 이진화
binary_preds = [(df["generated"] >= 0.5).astype(int) for df in dfs]

# 각 샘플별로 1이 나온 비율 → 확률처럼 계산
num_models = len(binary_preds)
soft_hard_probs = [
    sum(preds[i] for preds in binary_preds) / num_models
    for i in range(len(binary_preds[0]))
]

# 결과 저장
final_submission = dfs[0][["ID"]].copy()
final_submission["generated"] = soft_hard_probs  # 0~1 사이 확률

# 저장
final_submission.to_csv("soft_hard_voting_submission.csv", index=False)
print("✅ Soft-Hard Voting 결과 저장 완료: soft_hard_voting_submission.csv")
