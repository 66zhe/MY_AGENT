import streamlit as st
import os
from dotenv import load_dotenv
from zhipuai import ZhipuAI

load_dotenv()

# 导入我们写好的单例组件
from rag.rag_pipeline import rag_answer
from rag.chunker import smart_chunker
from utils.ragstorage import get_chroma, COLLECTION_NAME

EMBED_MODEL = "embedding-3"

st.set_page_config(
    page_title="MY_AGENT RAG 知识库问答",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* 主标题样式 */
  .main > div:first-child > h1 {
    color: #1a5276;
    font-size: 28px;
    font-weight: 700;
  }
  /* 侧边栏标题 */
  .css-1d391kg > div > h2 {
    color: #1a5276;
    font-size: 20px;
  }
  /* 底部 Powered by */
  .footer {
    text-align: center;
    color: #7f8c8d;
    font-size: 12px;
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #d5d8dc;
  }
  /* 上传区域美化 */
  .stFileUploader > div {
    border: 2px dashed #2980b9;
    border-radius: 10px;
    padding: 20px;  /* ✅ 冒号 */
  }
</style>
""", unsafe_allow_html=True)

st.title("📚 MY_AGENT · RAG 知识库问答助手")
st.caption("基于智谱 GLM (glm-4-flash) + ChromaDB 的本地知识库问答系统")

def _get_client():
    api_key = os.getenv("ZHIPUAI_API_KEY")
    if not api_key:
        raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")
    return ZhipuAI(api_key=api_key)

def get_embedding(text, client):
    resp = client.embeddings.create(model=EMBED_MODEL, input=[text], dimensions=1024)
    return resp.data[0].embedding

def ingest_text(text, file_name):
    chunks = smart_chunker(text, chunk_size=400, overlap=80)
    if not chunks:
        raise RuntimeError("smart_chunker 没有返回文本块")

    client = _get_client()
    embeddings = [get_embedding(c, client) for c in chunks]

    collection = get_chroma().get_or_create_collection(COLLECTION_NAME)
    ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": file_name, "index": i} for i in range(len(chunks))]
    
    collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)
    return len(chunks)

def kb_stats():
    collection = get_chroma().get_or_create_collection(COLLECTION_NAME)
    docs = collection.get()
    total_chunks = len(docs["ids"])
    sources = {m["source"] for m in docs["metadatas"] if m and "source" in m}
    return len(sources), total_chunks

with st.sidebar:
    st.header("知识库管理")

    st.markdown("### 上传文档")
    st.caption("支持 .txt 格式, 上传后自动分块、向量化、入库")
    uploaded = st.file_uploader("选择文件", type="txt", label_visibility="collapsed")
    if uploaded is not None:
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="ignore")
            n = ingest_text(raw, uploaded.name)
            st.success(f"✅️ 已入库 《{uploaded.name}》, 新增 {n} 个文本块")
        except Exception as e:
            st.error(f"❌️ 入库失败: {e}")

    st.markdown("---")
    num_docs, num_chunks = kb_stats()
    st.metric("📃 知识库文档数", num_docs)
    st.metric("🧩 文本块总数", num_chunks)

    st.markdown("---")
    if st.button("🗑️ 清空知识库", type="secondary"):
        try:
            get_chroma().delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("💡 使用示例")
    st.markdown("""
    **你可以这样提问: **
    1  上传一篇关于"量子计算"的文章，然后问：
    > 量子计算的核心概念是什么？

    2 上传一篇关于"AI发展"的文章，然后问：
    > AI目前有哪些主要应用领域?

    3 上传多篇文章，问某篇特有的内容：
    > 这篇文章提到了哪些缓解焦虑的方法？
    
    **提示：** 问题越具体，检索越精准。如果资料中没有相关内容，我会如实告诉你。
    """)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ 正确写法
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("请输入你的问题, 例如: 量子计算是什么? "):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("思考中...正在检索知识库并生成回答..."):
            answer = rag_answer(prompt)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown('<div class="footer">Powered by <strong>大哲</strong> · MY_AGENT RAG 知识库问答系统</div>', unsafe_allow_html=True)