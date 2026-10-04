import streamlit as st
import os
import yaml
from dotenv import load_dotenv
from zhipuai import ZhipuAI
load_dotenv()

# 导入我们写好的单例组件
from agent.chat_agent import ChatAgent
from agent.writer_agent import WriterAgent
from rag.rag_pipeline import rag_answer
from rag.chunker import smart_chunker
from utils.ragstorage import get_chroma, COLLECTION_NAME

# 接入四个工具
from tools.weather_tool import get_weather
from tools.search_tool import search_web
from tools.calculator_tool import calculator
from tools.rag_tool import search_knowledge_base

EMBED_MODEL = "embedding-3"

def load_config(path="config/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

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
    padding: 20px;
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

# 解析 Researcher 结构化素材清单，宽松匹配，解析失败退回默认，绝不报错（呼应 Day50 不崩原则）
def parse_sources(raw: str) -> str:
    sources = []
    try:
        kb_seg = raw.split("【本地知识库】")[-1].split("【")[0] if "【本地知识库】" in raw else ""
        if kb_seg.strip() and not kb_seg.strip().startswith("无"):
            sources.append("本地知识库")
        web_seg = raw.split("【实时网络】")[-1].split("【")[0] if "【实时网络】" in raw else ""
        if web_seg.strip() and not web_seg.strip().startswith("无"):
            sources.append("实时网络")
        calc_seg = raw.split("【数据/换算】")[-1].split("【")[0] if "【数据/换算】" in raw else ""
        if calc_seg.strip() and not calc_seg.strip().startswith("无"):
            sources.append("数据换算")
    except Exception:
        pass
    return " + ".join(sources) if sources else "多源素材"

with st.sidebar:
    st.header(" 功能模式")
    model = st.radio(
        "选择模式",
        ["🤖 智能对话 (工具 + 知识库)",
         "📚 纯知识库回答 (RAG)",
         "👥 双 Agent 协作 (研究报告)"],
         label_visibility="collapsed"
    )
    st.markdown("---")
    st.header("知识库管理")
    st.markdown("### 上传文档")
    st.caption("支持 .txt 格式, 上传后自动分块、向量化、入库")
    uploaded = st.file_uploader("选择文件", type="txt", label_visibility="collapsed")
    if uploaded is not None and uploaded.name not in st.session_state.get("ingested", set()):
        try:
            raw = uploaded.getvalue().decode("utf-8", errors="ignore")
            n = ingest_text(raw, uploaded.name)
            st.session_state.setdefault("ingested", set()).add(uploaded.name)
            st.success(f"已入库 《{uploaded.name}》, 新增 {n} 个文本块")
            st.rerun()
        except Exception as e:
            st.error(f"入库失败 {e}")
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
        st.session_state["ingested"] = set()
        st.rerun()
    st.markdown("---")
    st.header("💡 使用示例")
    st.markdown("""
    **你可以这样提问:**
    1  上传一篇关于"量子计算"的文章，然后问：
    > 量子计算的核心概念是什么？
    2 上传一篇关于"AI发展"的文章，然后问：
    > AI目前有哪些主要应用领域?
    3 上传多篇文章，问某篇特有的内容：
    > 这篇文章提到了哪些缓解焦虑的方法？

    **提示：** 问题越具体，检索越精准。如果资料中没有相关内容，我会如实告诉你。
    """)

@st.cache_resource
def build_agent():
    config = load_config()
    config['api_key'] = os.getenv("ZHIPUAI_API_KEY")
    agent = ChatAgent(config)
    agent.register_tool('get_weather', get_weather)
    agent.register_tool('search_web', search_web)
    agent.register_tool('calculator', calculator)
    agent.register_tool('search_knowledge_base', search_knowledge_base)
    return agent

@st.cache_resource
def build_writer():
    config = load_config()
    config['api_key'] = os.getenv("ZHIPUAI_API_KEY")
    return WriterAgent(config)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("请输入你的问题, 例如: 西安天气怎么样/量子计算的核心概念? "):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if model == "🤖 智能对话 (工具 + 知识库)":
        agent = build_agent()
        with st.chat_message("assistant"):
            with st.spinner("思考中...正在决定是否调用工具"):
                reply = agent.react(prompt)
        st.write(reply)

    elif model == "📚 纯知识库回答 (RAG)":
        with st.chat_message("assistant"):
            with st.spinner("检索知识库并生成回答..."):
                reply = rag_answer(prompt)
        st.write(reply)

    else:
        # 双 Agent 协作：改法一（结构化素材清单 + 优先知识库）+ 改法二（四段式重写 + 来源播报 + Markdown 渲染）
        agent = build_agent()
        writer = build_writer()
        research_instruction = (
            f"用户想深入了解关于【{prompt}】的内容，需要写一篇科技科普报告。\n"
            "请你作为 Researcher，调用工具尽可能全面地搜集素材，并按下面的格式汇报：\n"
            "【本地知识库】<search_knowledge_base 命中的要点，若为空写'无'>\n"
            "【实时网络】<search_web 补充的要点，若无写'无'>\n"
            "【数据/换算】<calculator 结果，若无写'无'>\n"
            "要求：只汇报工具原始素材，不要自己写文章、不要回答用户。"
        )
        with st.chat_message("assistant"):
            with st.spinner("🕵️ Researcher 搜集素材…"):
                raw = agent.chat(research_instruction)        # 对齐 main.py: researcher.chat(...)

            sources = parse_sources(raw)
            st.caption(f"🧾 本次报告素材来源：{sources} ｜ 以下为 Writer 重写的四段式报告")

            with st.spinner("✍️ Writer 撰写四段式报告…"):
                reply = writer.write_article(prompt, raw)     # 对齐 main.py
            st.markdown(reply)                              # 四段式带 #/## 标题，用 markdown 才渲染层级

    st.session_state.messages.append({"role": "assistant", "content": reply})

st.markdown('<div class="footer">Powered by <strong>大哲</strong> · MY_AGENT 智能助手</div>', unsafe_allow_html=True) 