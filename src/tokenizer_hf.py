from transformers import AutoTokenizer


# Hugging Face 모델명 기반 토크나이저 로드
def get_tokenizer(model_name, use_fast=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast)
    return tokenizer
