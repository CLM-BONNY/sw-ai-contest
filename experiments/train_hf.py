import yaml
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets import Dataset

from src.tokenizer_hf import get_tokenizer
from src.model_hf import get_model
from src.trainer_hf import train_model
from utils.seed import set_seed
from utils.logger import get_logger, log_metrics

# config 파일 내부 설정 로딩
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

project_name = config["project_name"]
model_name = config["model_name"]
train_path = config["train_path"]
save_path = config["save_path"]
max_length = config["max_length"]
batch_size = config["batch_size"]
epochs = config["epochs"]
seed = config["seed"]
num_labels = config["num_labels"]
lr = config["lr"]
weight_decay = config["weight_decay"]

# run name 설정
run_name = f"{model_name}-lr{lr}-bs{batch_size}-ep{epochs}"

# 로거 설정
logger = get_logger("HF-Train")

# W&B 설정
wandb.init(
    project=project_name,
    name=run_name,
    config={
        "model": model_name,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "max_length": max_length,
        "seed": seed,
    },
)

# 시드 고정 및 로깅 설정
set_seed(seed)
logger = get_logger("HF-Train")

# 데이터 로딩 및 전처리
df = pd.read_csv(train_path)
df = df.rename(columns={"paragraph_text": "full_text"})  # 열 이름 통일

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["generated"], random_state=seed
)

# 토크나이저 로딩
tokenizer = get_tokenizer(model_name)


def tokenize(example):
    return tokenizer(
        example["full_text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("generated", "labels")
val_dataset = val_dataset.rename_column("generated", "labels")
train_dataset.set_format("torch")
val_dataset.set_format("torch")


# 모델 로딩 설정
model = get_model(model_name, num_labels)

trainer = train_model(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    output_dir=save_path,
    config={
        "train_batch_size": batch_size,
        "eval_batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
    },
)

# 학습 및 모델 저장
logger.info("Start training...")
trainer.train()

logger.info("Saving model...")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

# 최종 성능 평가 및 로깅 (W&B + 콘솔 + 파일)
final_metrics = trainer.evaluate()
log_metrics(logger, final_metrics)

logger.info("Training completed.")

wandb.finish()
