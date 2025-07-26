import os
import gdown
import pandas as pd
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 환경 변수에서 OpenAI API 키 불러오기
# ==========================================
API_KEY = os.environ.get("OPENAI_API_KEY")
if not API_KEY:
    st.error("OpenAI API Key가 설정되지 않았습니다. Streamlit Secrets 또는 환경 변수에서 OPENAI_API_KEY를 설정하세요.")
    st.stop()

client = OpenAI(api_key=API_KEY)

# ==========================================
# 2. 구글 드라이브에서 데이터 파일 다운로드
# ==========================================
INDEX_FILE_ID = "10O9D9kIHHbRPMJN_52mPCSjYklgBZpMJ"  # 드라이브에서 추출한 파일 ID
CSV_FILE_ID = "1HpNAK0vO11XiifJexX7t3Ly902WSGKEJ"      # 드라이브에서 추출한 파일 ID

index_file = "slowletter_entities.index"
csv_file = "slowletter_full_with_entities.csv"

if not os.path.exists(index_file):
    st.write("Index 파일을 다운로드 중입니다...")
    gdown.download(f"https://drive.google.com/uc?id={INDEX_FILE_ID}", index_file, quiet=False)

if not os.path.exists(csv_file):
    st.write("CSV 파일을 다운로드 중입니다...")
    gdown.download(f"https://drive.google.com/uc?id={CSV_FILE_ID}", csv_file, quiet=False)

# ==========================================
# 3. 데이터와 벡터 인덱스 로드 (캐싱)
# ==========================================
@st.cache_resource
def load_resources():
    index = faiss.read_index(index_file)
    df = pd.read_csv(csv_file)
    return index, df

index, df = load_resources()

# ==========================================
# 4. 검색 함수
# ==========================================
def search_answer(query, top_k=5):
    # 질문 임베딩
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding
    emb = np.array(emb).astype("float32").reshape(1, -1)

    # 벡터 검색
    D, I = index.search(emb, top_k)
    rows = df.iloc[I[0]]

    # 검색 컨텍스트 생성
    context = "\n\n".join(
        f"{r.get('date','')}\n제목: {r.get('h3_title','')}\n본문: {r.get('h3_content_html','')}\n엔티티:{r.get('entities','')}\n이벤트:{r.get('events','')}"
        for _, r in rows.iterrows()
    )

    # GPT 답변 생성
    prompt = f"다음 컨텍스트를 참고해 질문에 답해 주세요.\n\n질문:\n{query}\n\n컨텍스트:\n{context}"

    ans = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}]
    ).choices[0].message.content

    return ans

# ==========================================
# 5. Streamlit UI
# ==========================================
st.title("Slowletter Q&A Demo")

query = st.text_area("질문을 입력하세요:")

if st.button("검색"):
    if query.strip():
        with st.spinner("검색 중..."):
            answer = search_answer(query)
            st.write("### 답변")
            st.write(answer)
