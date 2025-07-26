import streamlit as st
import pandas as pd
import numpy as np
import faiss
from openai import OpenAI
import os

# ===== 설정 =====

import os
API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)
INDEX_PATH = r"\slowletter_entities.index"
CSV_PATH = r"\slowletter_full_with_entities.csv"

client = OpenAI(api_key=API_KEY)

@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_PATH)
    df = pd.read_csv(CSV_PATH)
    return index, df

index, df = load_resources()

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

    # 컨텍스트 생성
    context = "\n\n".join(
        f"{r['date']}\n제목: {r['h3_title']}\n본문: {r['h3_content_html']}\n엔티티:{r.get('entities','')}\n이벤트:{r.get('events','')}"
        for _, r in rows.iterrows()
    )

    # GPT에게 질문
    prompt = f"""
다음 컨텍스트를 참고해 질문에 답해 주세요.

질문:
{query}

컨텍스트:
{context}
"""
    ans = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}]
    )
    return ans.choices[0].message.content

st.title("Slowletter Q&A MVP")
query = st.text_area("질문을 입력하세요:")

if st.button("검색"):
    if query.strip():
        with st.spinner("검색 중..."):
            answer = search_answer(query)
            st.write("### 답변")
            st.write(answer)