# 2025 SW중심대학 디지털 경진대회 : AI부문
## 생성형 AI(LLM)와 인간 : 텍스트 판별 챌린지

## 1 프로젝트 개요

**1.1 개요**

 본 대회는 ‘생성형 AI(LLM)와 인간 : 텍스트 판별 챌린지’라는 주제로 진행되며 문단 단위(Paragraph)의 글(Text)이 사람(Human)이 작성한 것(0)인지, 생성 AI가 작성한 것(1)인지 판별하여, 각 문단(샘플)이 AI가 작성했을 확률을 예측하는 AI 모델을 개발하는 태스크를 부여한다.  본 문제에 사용되는 데이터는 대부분 한국어(Korean)로 구성된 텍스트 데이터이며 학습 데이터는 전체 글(Full Text)에 대해, 일부 문단이나 문장만 AI가 작성된 경우에도 글 전체에 'AI 작성' 라벨(1)이 부여되며, 문단 단위 라벨은 제공되지 않는다. 

 평가 데이터는 1개 문단 단위로 샘플이 구성되어 있으며, 샘플이 AI가 작성했을 확률(0~1 사이)을 예측해야 한다. 평가 데이터에는 `title`과 `paragraph_index` 정보가 포함되어 있으며, 같은 `title`을 가진 문단들은 하나의 글에 속하는 문단들이므로, 이를 바탕으로 동일 글 내의 다른 문단 정보를 추론에 활용하는 것이 허용된다. 이는 일반적인 평가 환경에서의 데이터 누수(Data Leakage)와는 다른 구조로, 하나의 글 내 문단 간 상호 참조는 허용되며, 서로 다른 글 간에는 정보가 공유되지 않도록 해야 한다.

<br />
<br />

**1.2 평가지표**

평가 지표는 ROC-AUC이다. 여기서 TPR (True Positive Rate)은 참 양성 비율, 즉 실제 양성 중에서 모델이 양성이라고 맞게 예측한 비율, FPR (False Positive Rate)은 위양성률, 즉 실제 음성인데 모델이 양성으로 잘못 예측한 비율이다.
<img width="600" alt="ROC-AUC" src="https://github.com/user-attachments/assets/08d13e89-17f6-4363-a8ab-11ce4e0942a5" />

<br />
<br />

## 2 프로젝트 팀 구성 및 역할

## 팀원 소개
| 이름 | 역할 | GitHub |
| --- | --- | --- |
| 김민선 | - 팀장 <br /> - EDA <br /> - 사전 학습 모델 선택 실험 <br /> - 데이터 증강: 라벨 1 증강 비율 실험 <br /> - 데이터 샘플링: 전체 데이터 비율 유지 후 데이터 추출 <br /> - LoRA 적용 파이프라인 구축 <br /> - 앙상블 실험 <br /> - 프로젝트 템플릿 구축 <br /> - Streamlit Data Viewer 제작 | [CLM-BONNY](https://github.com/CLM-BONNY) |
| 최건웅 | - EDA <br /> - 사전 학습 모델 선택 실험 <br /> - 데이터 전처리: 연속 공백 / 유니코드 제어문자 제거 <br /> - 데이터 증강: 라벨 1 학습 모델로 증강 데이터 생성 <br /> - 형태소 분석 실험 <br /> - 허깅페이스 모델 성능 개선 실험 | [ddugel3](https://github.com/ddugel3) |
| 황솔희 | - EDA <br /> - 사전 학습 모델 선택 실험 <br /> - 데이터 전처리: 극단적으로 긴 데이터 제거 <br /> - 데이터 증강: Back Translation <br /> - 허깅페이스 모델 성능 개선 실험 | [ssoree912](https://github.com/ssoree912)

<br />
<br />

## 3 프로젝트

**3.1 프로젝트 진행 일정**

- EDA, 데이터 전처리, 데이터 증강, 데이터 샘플링, 사전학습 모델 검증 및 성능 향상 실험, LoRA 적용 실험, 앙상블 순서로 진행
<img width="1200" alt="프로젝트 진행일정" src="https://github.com/user-attachments/assets/99f55652-4eb4-4a0e-ba7c-bb9ca11dc769" />

<br />
<br />

**3.2 코드 구조**

```bash
# 프로젝트 디렉토리 구조 및 설명

Template/
├── config/                                       # 공통 설정 파일 보관 폴더
│   └── config.yaml                               # 실험별 모델, 데이터 경로 및 하이퍼파라미터 설정
│
├── data/                                         # 데이터 저장 폴더
│   ├── raw/                                      # 원본 CSV 파일 (train.csv, test.csv 등)
│   ├── processed/                                # 전처리된 데이터 파일 저장 폴더
│   └── augmented/                                # 증강된 데이터 파일 저장 폴더
│
├── src/                                          # 주요 모듈 코드 저장 폴더
│   ├── __init__.py                               # 패키지 인식용 파일
│   ├── ensemble/                                 # 앙상블 관련 코드 폴더
│   ├── vectorizer.py                             # Scikit-learn 기반 벡터화 로직 (TF-IDF)
│   ├── tokenizer_hf.py                           # Hugging Face AutoTokenizer 래퍼 함수 정의
│   ├── model_ml.py                               # XGBoost 등 전통 ML 모델 정의 클래스
│   ├── model_hf.py                               # Hugging Face 사전학습 모델 래퍼 클래스
│   ├── model_hf_LoRA.py                          # Hugging Face 모델 + LoRA 적용 래퍼 클래스
│   ├── trainer_ml.py                             # 전통 ML 모델 학습 루프 정의 (XGBoost)
│   ├── trainer_hf.py                             # Hugging Face Trainer 설정 및 학습 루프 정의
│   ├── backtranslation.py                        # 데이터 증강용 백번역 함수 정의
│   ├── text_cleaning.py                          # 텍스트 전처리 (정규화, 특수문자 제거 등) 함수 정의
│   └── evaluator.py                              # 모델 평가 지표 계산 함수 (ROC-AUC, F1 등)
│
├── experiments/                                  # 학습 및 추론 실행 스크립트 폴더
│   ├── __init__.py                               # 패키지 인식용 파일
│   ├── train_baseline.py                         # TF-IDF + XGBoost 베이스라인 학습 실행 스크립트
│   ├── train_hf.py                               # Hugging Face 모델 학습 실행 스크립트
│   ├── train_hf_LoRA.py                          # Hugging Face 모델 학습 + LoRA 적용 실행 스크립트
│   ├── infer_hf.py                               # Hugging Face 모델 로드 후 테스트셋 예측 수행
│   └── infer_hf_LoRA.py                          # LoRA 적용 Hugging Face 모델 로드 후 테스트셋 예측 수행
│
├── models/                                       # 학습된 모델 및 체크포인트 저장 폴더
│   └── hf_checkpoint/                            # Hugging Face 모델 체크포인트 저장 경로
│
├── utils/                                        # 공통 유틸리티 함수 모음 폴더
│   ├── __init__.py                               # 패키지 인식용 파일
│   ├── logger.py                                 # 로깅 포맷 정의 및 로그 저장 함수
│   └── seed.py                                   # 랜덤 시드 고정 함수
│
├── app/                                          # Streamlit 기반 시각화 대시보드
│   └── data_viewer.py                            # Streamlit 앱에서 데이터 테이블 시각화 코드
│
├── notebooks/                                    # 실험 분석 및 결과 정리용 Jupyter 노트북 모음
│   ├── baseline.ipynb                            # TF-IDF + XGBoost 베이스라인 실험 노트북
│   ├── EDA_minseon.ipynb                         # 민선 EDA 분석 노트북
│   ├── EDA_geonug.ipynb                          # 건웅 EDA 분석 노트북
│   ├── EDA_solhee.ipynb                          # 솔희 EDA 분석 노트북
│   ├── extreme_long.ipynb                        # 긴 문장 필터링/처리 실험 노트북
│   ├── Best_TFIDF_XGBoost_Optimized.ipynb        # TF-IDF + XGBoost 성능 최적화 실험 노트북
│   └── morphs_tfidf_xgboost.ipynb                # 형태소 기반 TF-IDF + XGBoost 실험 노트북
│
├── docs/                                         # 프로젝트 실험 보고서 폴더
├── requirements.txt                              # 프로젝트 실행을 위한 패키지 의존성 목록
├── README.md                                     # 프로젝트 설명 문서
├── run_ml.sh                                     # 전체 학습-평가-추론 파이프라인 실행 스크립트 (XGBoost 모델)
└── run.sh                                        # 전체 학습-평가-추론 파이프라인 실행 스크립트 (HuggingFace 모델)
```

<br />
<br />

## 4 EDA

**4.1 클래스 비율**

 전체 학습 데이터는 약 97,000개이며 라벨 0(Human)이 89,177개, 라벨 1(AI)이 7,995개로 약 11:1 수준의 심각한 클래스 불균형을 보인다.
이로 인해 모델이 사람 글(0)에 과도하게 편향되어 학습될 수 있으며 극단적인 경우 “항상 사람”이라고만 예측해도 약 92%의 정확도를 기록할 수 있다.
이러한 분포는 학습에 부정적인 영향을 미칠 수 있으므로 라벨 간 데이터 균형을 맞추는 사전 작업이 필요하다는 인사이트를 얻었다.

<img width="400" alt="클래스 비율" src="https://github.com/user-attachments/assets/f9d1f1e5-5d01-4322-b9f5-44b95f17aba7" />

<br />
<br />

**4.2 텍스트 길이 & 문단 수 분석**

 텍스트 길이와 문단 수 분포를 확인한 결과, 대부분의 데이터는 텍스트 길이 20,000자 이하, 문단 수 50개 이하로 확인되었다. 전체 분포는 long-tail 형태를 띠며 극단적으로 긴 데이터가 일부 존재했는데 이는 학습 과정에서 잡음으로 작용할 가능성이 있다.
따라서 전처리 단계에서 텍스트가 과도하게 긴 샘플은 제거하거나 별도로 처리할 필요가 있다고 판단했다.

<div style={"display": "flex"; "justify-content": "center";}>
  <img width="400" alt="텍스트 길이" src="https://github.com/user-attachments/assets/80426ace-fa90-4a16-a8d2-410866006c32" />
  <img width="400" alt="문단 수 분석" src="https://github.com/user-attachments/assets/0b30f45a-fb2a-4f15-8392-7f00291165cc" />
</div>

<br />
<br />

**4.3 토큰 길이 분석**

 `klue/bert-base` 토크나이저 기준으로 토큰 길이를 분석한 결과, 512 토큰을 초과하는 샘플이 다수 존재했으며, 최장 길이는 55,143 토큰까지 나타났다.
이로 인해 학습 시에는 토큰 분할이 필수적이며 sliding window, truncation, stride 등 적절한 전처리 전략이 필요하다.

<img width="800" alt="토큰 길이 분석" src="https://github.com/user-attachments/assets/1eafd01c-7ae1-4717-9c33-6df6af6c6be2" />

<br />
<br />

**4.4 어휘 다양성 (TTR)**

 “LLM이 작성한 글은 어휘가 반복될 것이다”라는 가설을 세우고 TTR(Type-Token Ratio)을 기준으로 분석을 진행했다. 하지만 실제 결과는 예상과 달랐다. AI 문장(1)의 TTR 중앙값이 사람 문장(0)보다 오히려 약간 높게 나타났다.
이는 최근 LLM의 표현력이 인간 수준에 도달하거나 일부 영역에서는 초과할 수도 있음을 시사하는 흥미로운 결과였다.

<img width="800" alt="어휘 다양성" src="https://github.com/user-attachments/assets/88f5e459-2ed0-41f7-a07b-c4821d79ae49" />

<br />
<br />

**4.5 첫 문장 주어 유무**

 데이터 탐색 중 AI가 쓴 글에서 주어가 생략된 문장이 처음에 오는 경우가 많다는 인상을 받아 이를 정량적으로 분석해보았다. 하지만 실제로 분석한 결과, 사람 글과 AI 글 모두 첫 문장에 주어가 없는 비율이 비슷해 의미 있는 차이는 발견되지 않았다.
직관과는 달랐던 분석 결과였지만 직접 가설을 세우고 확인했다는 점에서 의미 있는 시도였다.

<img width="800" alt="첫 문장 주어 유무" src="https://github.com/user-attachments/assets/ab2488a6-455c-4bb7-b55c-b820f1316c76" />

<br />
<br />

**4.6 쉼표 개수**

 LLM이 생성한 문장에서는 쉼표 사용이 많을 것이라는 가설을 바탕으로 문장당 쉼표 평균 개수를 라벨별로 분석했다.
사람 문장(0)은 평균 0.72개, AI 문장(1)은 평균 0.41개로 사람 문장이 쉼표를 더 많이 사용하는 것으로 나타났다. 특히 사람 문장은 이상치가 많고 분포 자체가 훨씬 넓은 경향을 보였다. 상위 25% 구간(Q3) 기준에서도 사람 문장이 쉼표를 더 자주 사용하는 등 쉼표 사용량 차이는 꽤 뚜렷했다.
기존에 알고 있던 LLM의 문체 특성과는 다소 다른 결과였으며 사람 문장이 한 문장당 더 많은 정보를 담는 구조이기 때문에 이런 차이가 발생했을 수도 있다는 생각이 들었다.

<img width="800" alt="쉼표 개수" src="https://github.com/user-attachments/assets/c1d7172d-2e77-4459-8186-642c61317ef5" />

<br />
<br />

## 5 프로젝트 수행

**5.1 Data Processing**

- Data Preprocessing
    - 극단적으로 긴 데이터 제거
    - 연속 공백 / 유니코드 제어문자 제거
- Data Augmentation
    - 라벨 1 증강 비율 실험
    - 역번역(Back Translation)
    - 라벨 1 학습 모델을 활용한 생성 데이터 추가

**5.2** Modeling

- 형태소 분석
- 허깅페이스 모델 성능 개선
- LoRA + 데이터 샘플링 적용

**5.3 Ensemble**

- Weighted Voting

<br />
<br />

## 6 코드 실행 방법

```bash
# 라이브러리 설치
pip install -r requirements.txt

# Streamlit 데이터 뷰어 실행
streamlit run app/data_viewer.py

# ml 모델 학습 코드 실행
bash run_ml.sh

# 허깅페이스 모델 학습 코드 실행
bash run.sh
```

<br />
<br />


## 7 실험 보고서

자세한 내용은 <a href="https://github.com/CLM-BONNY/sw-ai-contest/blob/main/docs/2025%20SW%EC%A4%91%EC%8B%AC%EB%8C%80%ED%95%99%20%EB%94%94%EC%A7%80%ED%84%B8%20%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C_AI%EB%B6%80%EB%AC%B8_%EC%8B%A4%ED%97%98%EB%B3%B4%EA%B3%A0%EC%84%9C%20(203%ED%98%B8).pdf">실험 보고서</a>를 참고해 주세요 !
