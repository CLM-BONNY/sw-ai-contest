from transformers import Trainer, TrainingArguments
from src.evaluator import compute_metrics
from utils.logger import get_logger

logger = get_logger(__name__)


# Hugging Face Trainer 활용 모델 학습 함수
def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir="models/hf_checkpoint",
    config=None,
):
    # 학습 관련 설정 정의
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        per_device_train_batch_size=config.get("train_batch_size", 8),
        per_device_eval_batch_size=config.get("eval_batch_size", 8),
        num_train_epochs=config.get("epochs", 3),
        learning_rate=config.get("lr", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",  # wandb 사용 시 'wandb'
    )

    # Hugging Face Trainer 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 모델 학습 수행 및 로그 출력
    logger.info("모델 학습 시작")
    trainer.train()
    logger.info("모델 학습 완료")

    # 학습 완료된 Trainer 객체 반환
    return trainer
