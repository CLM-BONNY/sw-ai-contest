import yaml
import pandas as pd
import torch
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset

from utils.logger import get_logger
from utils.seed import set_seed


# config 파일 내부 설정 로딩
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_path = config["save_path"]
test_path = config["test_path"]
max_length = config["max_length"]
output_path = config.get("predict_path", "predictions.csv")
seed = config["seed"]


# 시드 고정 및 로깅 설정
set_seed(seed)
logger = get_logger("HF-Infer")

# 데이터 로딩 및 전처리
df = pd.read_csv(test_path)
df = df.rename(columns={"paragraph_text": "full_text"})  # 열 이름 통일

# 토크나이저 로딩
tokenizer = AutoTokenizer.from_pretrained(model_path)


def tokenize(example):
    return tokenizer(
        example["full_text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


test_dataset = Dataset.from_pandas(df)
test_dataset = test_dataset.map(tokenize, batched=True)
test_dataset.set_format("torch")

# 모델 로딩 및 추론
model = AutoModelForSequenceClassification.from_pretrained(model_path)
trainer = Trainer(model=model, tokenizer=tokenizer)

logger.info("Start inference...")
preds = trainer.predict(test_dataset)

# 예측 결과 후처리
if model.config.num_labels == 1:
    # 출력이 1개인 이진 분류 (시그모이드 확률 → 0.5 기준으로 분류)
    pred_labels = (
        (torch.sigmoid(torch.tensor(preds.predictions)) > 0.5).int().numpy().flatten()
    )
else:
    # 출력이 여러 개인 경우 (다중 분류 또는 로짓 2개짜리 이진 분류)
    # 가장 큰 값의 인덱스를 예측 클래스 레이블로 사용
    pred_labels = torch.argmax(torch.tensor(preds.predictions), dim=1).numpy()

# 추론 결과 저장
submission = pd.DataFrame({"ID": df["ID"], "generated": pred_labels})
submission.to_csv(output_path, index=False)
logger.info(f"Prediction saved to {output_path}")
