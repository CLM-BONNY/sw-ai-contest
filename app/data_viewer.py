import streamlit as st
import pandas as pd
import os

# train 데이터 로드, 캐싱
DATA_PATH = os.path.join("data", "raw", "train.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# 사이드바 - 라벨 필터
st.sidebar.header("🎯 라벨 필터")
selected_label = st.sidebar.selectbox("Generated Label 선택", options=["전체", "0", "1"])

if selected_label == "전체":
    filtered_df = df
else:
    filtered_df = df[df["generated"] == int(selected_label)]

index_list = filtered_df.index.tolist()  # 원본 인덱스 리스트
max_pos = len(index_list) - 1

# 세션 상태 초기화
if "idx_pos" not in st.session_state:
    st.session_state.idx_pos = 0

def change_pos(new_pos: int):
    st.session_state.idx_pos = max(0, min(new_pos, max_pos))

# train 데이터 full_text 뷰어 UI
st.title("📚 train 데이터 살펴보기")

# 드롭다운으로 인덱스 선택
select_idx = st.selectbox(
    label=f"확인할 인덱스 선택 (전체 {len(index_list):,}건)",
    options=index_list,
    index=st.session_state.idx_pos if st.session_state.idx_pos <= max_pos else 0
)
# 선택 즉시 위치 반영
if select_idx != index_list[st.session_state.idx_pos]:
    st.session_state.idx_pos = index_list.index(select_idx)

# 이전/다음 버튼 가운데 정렬
col_left, col_mid, col_right = st.columns([1,1,1])

with col_mid:   # 가운데 열에 두 버튼 수평 배치
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ 이전"):
            change_pos(st.session_state.idx_pos - 1)
    with col_next:
        if st.button("➡️ 다음"):
            change_pos(st.session_state.idx_pos + 1)

# 본문 표시
if len(filtered_df) == 0:
    st.warning("선택한 조건에 해당하는 데이터가 없습니다.")
else:
    # 현재 위치의 실제 행 가져오기
    current_idx = index_list[st.session_state.idx_pos]
    row = df.loc[current_idx]

    # Index + Label 뱃지
    col1, _ = st.columns([6, 1])
    with col1:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; gap: 10px;'>
                <h3 style='margin: 0;'>Index {current_idx}</h3>
                <div style='
                    background-color:{"#00BFFF" if row["generated"] == 0 else "#FF6347"};
                    color:white;
                    padding:4px 10px;
                    border-radius:8px;
                    font-size:0.85rem;
                '>Label {row['generated']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.write(row["full_text"])
