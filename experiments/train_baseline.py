import time
import pandas as pd
from sklearn.model_selection import train_test_split

from src.vectorizer import get_vectorizer
from src.model_ml import build_model
from src.trainer_ml import train_model, evaluate_model, predict

# 데이터 로딩
train = pd.read_csv('data/raw/train.csv', encoding='utf-8-sig')
test = pd.read_csv('data/raw/test.csv', encoding='utf-8-sig')

# 라벨과 입력 분리
X = train[['title', 'full_text']]
y = train['generated']
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# 벡터화
print("벡터화 시작")
start_time = time.time()
vectorizer = get_vectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
print(f"소요 시간: {time.time() - start_time:.2f}초")

# 모델 학습 및 평가
print("모델 학습 및 평가 시작")
model = build_model()
model = train_model(model, X_train_vec, y_train)
evaluate_model(model, X_val_vec, y_val)
print("모델 학습 및 평가 완료")

# 테스트셋 전처리 및 예측
print("테스트셋 전처리 및 예측 시작")
test = test.rename(columns={'paragraph_text': 'full_text'})
X_test = test[['title', 'full_text']]
X_test_vec = vectorizer.transform(X_test)
test_probs = predict(model, X_test_vec)
print("테스트셋 전처리 및 예측 완료")

# 결과 저장
print("결과 저장 시작")
sample_submission = pd.read_csv('data/sample_submission.csv', encoding='utf-8-sig')
sample_submission['generated'] = test_probs
sample_submission.to_csv('data/raw/baseline_submission.csv', index=False)
print("결과 저장 완료")
