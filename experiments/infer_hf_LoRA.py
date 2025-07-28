import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml
import wandb
import pandas as pd
import torch
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from peft import PeftModel, PeftConfig

from utils.logger import get_logger
from utils.seed import set_seed

# config 로딩
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

project_name = config["project_name"]
model_path = "models/hf_adapter/checkpoint-48590"
test_path = config["test_path"]
max_length = config["max_length"]
output_path = config["predict_path"]
seed = config["seed"]

run_name = f"infer-{model_path.split('/')[-1]}"

# W&B 설정
wandb.init(
    project=project_name,
    name=run_name,
    config={
        "model_path": model_path,
        "max_length": max_length,
        "test_path": test_path,
        "output_path": output_path,
    },
)

set_seed(seed)
logger = get_logger("HF-Infer")

# 데이터 로딩
df = pd.read_csv(test_path)
df = df.rename(columns={"paragraph_text": "full_text"})

# 어댑터 있는지 확인
adapter_config_path = os.path.join(model_path, "adapter_config.json")
if os.path.exists(adapter_config_path):
    # 어댑터 로딩
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_name = peft_config.base_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(base_model, model_path)
    logger.info("LoRA 어댑터 적용 모델 로딩 완료")
else:
    # 일반 모델 로딩
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    logger.info("기본 모델만 로딩")

model.eval()


# 토크나이즈
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

# 추론
trainer = Trainer(model=model, tokenizer=tokenizer)
logger.info("Start inference...")
preds = trainer.predict(test_dataset)

# 후처리
if model.config.num_labels == 1:
    # 시그모이드 확률 자체를 저장 (0~1 사이 값)
    pred_probs = torch.sigmoid(torch.tensor(preds.predictions)).numpy().flatten()
    pred_labels = pred_probs  # 그대로 저장
else:
    # 다중 분류는 소프트맥스 확률 중 최대값
    pred_probs = torch.softmax(torch.tensor(preds.predictions), dim=1).numpy()
    pred_labels = pred_probs.max(axis=1)  # 가장 큰 softmax 확률
wandb.finish()

# 결과 저장
print("결과 저장 시작")
sample_submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
sample_submission["generated"] = pred_labels
sample_submission.to_csv(output_path, index=False)
print("결과 저장 완료")
