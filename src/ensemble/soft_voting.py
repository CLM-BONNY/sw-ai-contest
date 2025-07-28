import pandas as pd
from functools import reduce

# 예측 결과 파일 경로 리스트
files = [
    "submission_koelectra_v3_discriminator.csv",
    "submission_KcELECTRA_2022.csv",
    "submission_klue_roberta.csv",
    "submission_daekeun-ml_koelectra-small-v3-nsmc.csv",
]

# 각 파일에서 'generated' 확률만 추출
dfs = [pd.read_csv(file) for file in files]
generated_probs = [df[["generated"]] for df in dfs]

# 확률 평균 계산 (Soft Voting)
average_probs = reduce(lambda x, y: x + y, generated_probs) / len(generated_probs)

# 최종 결과 생성
final_submission = dfs[0][["ID"]].copy()
final_submission["generated"] = average_probs

# 결과 저장
final_submission.to_csv("soft_voting_submission.csv", index=False)
print("✅ Soft voting 결과 저장 완료: soft_voting_submission.csv")
