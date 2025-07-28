import pandas as pd
from collections import Counter

# 예측 결과 파일 리스트
files = [
    "submission_koelectra_v3_discriminator.csv",
    "submission_KcELECTRA_2022.csv",
    "submission_klue_roberta.csv",
    "submission_daekeun-ml_koelectra-small-v3-nsmc.csv",
]

# 모든 파일에서 확률값 불러오기
dfs = [pd.read_csv(file) for file in files]

# 각 모델의 확률을 이진값(0 또는 1)으로 변환 (0.5 기준)
binary_preds = [(df["generated"] >= 0.5).astype(int) for df in dfs]

# 각 샘플별로 다수결 투표 수행
hard_voted = [
    Counter([preds[i] for preds in binary_preds]).most_common(1)[0][0]
    for i in range(len(binary_preds[0]))
]

# 결과 저장
final_submission = dfs[0][["ID"]].copy()  # ID는 첫 번째 파일 기준
final_submission["generated"] = hard_voted

# 파일로 저장
final_submission.to_csv("hard_voting_submission.csv", index=False)
print("✅ 하드보팅 결과 저장 완료: hard_voting_submission.csv")
