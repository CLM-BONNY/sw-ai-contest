from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import pandas as pd
import torch
from tqdm import tqdm

# 모델 로드
model_name = 'facebook/mbart-large-50-many-to-many-mmt'
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name).to('cuda')

# 번역 함수
def mbart_translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to('cuda')
    with torch.no_grad():
        generated_tokens = model.generate(
            **encoded,
            max_length=512,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang]
        )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# 데이터 로드
train = pd.read_csv('../data/raw/train.csv')

# generated == 1 데이터 필터
generated_1_df = train[train['generated'] == 1].copy().iloc[5001:]

augmented_data = []

# tqdm으로 진행 표시
for idx, row in tqdm(generated_1_df.iterrows(), total=generated_1_df.shape[0]):
    text = row['full_text']
    
    # None 또는 비어있는 값 방지방지
    if not isinstance(text, str) or not text.strip():
        print(f"⚠ Skipping index {idx}: empty or invalid text")
        continue
    
    try:
        # 한글 -> 영어 
        en_text = mbart_translate(text, "ko_KR", "en_XX")
        
        # 영어 -> 한글
        ko_back = mbart_translate(en_text, "en_XX", "ko_KR")
        
        augmented_data.append({
            'title': row['title'],
            'full_text': ko_back,
            'generated': row['generated']
        })
        
    except Exception as e:
        print(f"❌ Error at index {idx}: {e}")
        continue

# 증강 데이터프레임 생성
augmented_df = pd.DataFrame(augmented_data)

# 기존 데이터와 합치기
# final_df = pd.concat([train, augmented_df], ignore_index=True)

# 저장
augmented_df.to_csv('data/train_backtranslation5001_end.csv', index=False)

print(f"✅ 증강 완료! 최종 데이터 크기: {augmented_df.shape}")