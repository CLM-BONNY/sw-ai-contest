from transformers import AutoModelForSequenceClassification


# Hugging Face 모델 호출
def get_model(model_name, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model