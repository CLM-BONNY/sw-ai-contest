import pandas as pd
import re

# 데이터 불러오기
df = pd.read_csv("train.csv")

# 전처리 함수 정의
def clean_text(text):
    # 유니코드 제어문자 제거 (예: \x00, \u200b 등)
    text = re.sub(r'[\x00-\x1f\x7f-\x9f\u200b\u200c\u200d\uFEFF]', '', text)
    # 연속 공백을 하나로
    text = re.sub(r'\s+', ' ', text)
    # 양 끝 공백 제거
    return text.strip()

# 적용
df["full_text"] = df["full_text"].apply(clean_text)
df.to_csv("cleaned_train.csv", index=False)