from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForSequenceClassification

# Hugging Face 모델 호출 + LoRA 적용


def get_model(model_name, num_labels):
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # LoRA 설정
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    # LoRA 적용
    model = get_peft_model(base_model, lora_config)
    return model
