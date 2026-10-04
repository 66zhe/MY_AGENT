# MY_AGENT

> 一个跑在本地的多智能体 RAG 智能助手：支持文档上传问答、大模型自主工具调用（天气/计算/搜索/查知识库）、以及「Researcher 搜集 + Writer 撰写」的双 Agent 协作报告生成，全部集成在一个 Streamlit 网页里。

](https://python.org)
](https://streamlit.io)
](https://open.bigmodel.cn)


# 应用展示：
https://myagent-qfkvfxftczcl5zcfesbrhs.streamlit.app/

---

## 项目亮点

| 亮点 | 说明 |
|------|------|
| 多步链式工具调用 | 问"北京多少度？换算成华氏度"→ 自动走 `get_weather` + `calculator` 两步 |
| 双 Agent 协作 | Researcher 多源搜集 → Writer 重写四段式科技报告（标题/核心概念/应用场景/未来展望） |
| 工具容错不崩 | 调用失败自动重试 2 次 + 指数退避，仍失败优雅降级，程序永不崩溃 |
| 诚实不编造 | 知识库里没有的内容，如实告诉你"找不到"，不编造答案 |
| 多轮记忆 | 对话历史跨轮累积，不会"失忆" |

---

## 功能列表

### 一、本地 RAG 知识库问答
- 语义分块（`smart_chunker`，chunk_size=400，overlap=80）
- 智谱 `embedding-3`（1024 维）向量化
- ChromaDB 向量检索 + BM25 混合检索（`rank_bm25`），`top_k=3`
- 文档没有的内容**诚实说没有、不编造**

### 二、大模型自主工具调用（Function Calling + ReAct）
- `glm-4-flash` 自主判断是否需要调用工具，支持多步链式（ReAct 循环，`max_steps=5`）
- 四个工具：
  - `get_weather(city)` — 查询城市天气
  - `search_web(query)` — 模拟网络搜索
  - `calculator(expression)` — 数学计算
  - `search_knowledge_base(query)` — 检索本地知识库
- **容错机制**：`call_tool_with_retry(retries=2)` 自动重试 + 指数退避，失败则 `degraded_message` 优雅降级

### 三、双 Agent 协作（研究报告生成）
- **Researcher**（ChatAgent）：多源搜集素材（优先知识库 → 网络搜索 → 数据换算），按结构化格式汇报
- **Writer**（WriterAgent）：基于素材重写为四段式科技专栏（标题 / 核心概念 / 应用场景 / 未来展望）
- 界面实时播报"本次报告素材来源：本地知识库 + 实时网络"

### 四、多轮对话记忆
- 对话历史存在 `ChatAgent.self.messages` 中
- Streamlit 端通过 `@st.cache_resource` 缓存 Agent 实例，跨重跑保持记忆

### 五、Streamlit 三模式网页（Day51）
- 侧边栏一键切换：智能对话（工具 + 知识库）/ 纯知识库问答（RAG）/ 双 Agent 协作（研究报告）
- 支持上传 .txt 文件即时入库、知识库统计、一键清空

---

## 技术架构
![MY_AGENT技术架构图](docsarchitecture.png)

---

## 技术栈

| 层面 | 技术 |
|------|------|
| LLM | 智谱 GLM-4-Flash（`glm-4-flash`，temperature=0.3） |
| Embedding | 智谱 `embedding-3`（1024 维） |
| 向量数据库 | ChromaDB（本地持久化 `./chorma_db/`） |
| 混合检索 | 向量相似度 + BM25（`rank_bm25`） |
| Agent 框架 | 自研 ReAct 循环 + Function Calling |
| Web 前端 | Streamlit |
| 配置管理 | YAML（`config/config.yaml`）+ python-dotenv |

---

## 项目结构
my_agent/
├── app.py # Streamlit 主入口（三模式）
├── main.py # 命令行双 Agent 协作流水线
├── agent/
│ ├── chat_agent.py # ChatAgent：ReAct + 工具调用 + 记忆 + 容错
│ ├── writer_agent.py # WriterAgent：四段式写作
│ └── memory.py # 记忆模块
├── tools/
│ ├── weather_tool.py # get_weather(city)
│ ├── search_tool.py # search_web(query)
│ ├── calculator_tool.py # calculator(expression)
│ └── rag_tool.py # search_knowledge_base(query)
├── rag/
│ ├── chunker.py # smart_chunker 语义分块
│ ├── rag_embedding.py # embedding 封装
│ ├── search.py # BM25 + 向量混合检索
│ ├── rag_generator.py # 答案生成
│ ├── rag_pipeline.py # RAG 端到端流水线
│ ├── muti_doc_ingest.py # 多文档批量入库
│ └── article1-4.txt # 示例文档
├── utils/
│ └── ragstorage.py # ChromaDB 单例 + COLLECTION_NAME
├── config/
│ └── config.yaml # 模型/检索/prompt 配置
├── .streamlit/
│ └── config.toml # Streamlit 主题配置
├── .env # 环境变量（API Key，不提交）
├── .env.example # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md

---

## 快速开始

### 前置条件

- Python 3.10+
- 智谱开放平台账号（），获取 API Key

### 安装与运行

```bash
# 1. 克隆项目
git clone <https://github.com/66zhe/MY_AGENT.git> && cd my_agent

# 2. 创建虚拟环境（Windows）
python -m venv venv
venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
copy .env.example .env
# 编辑 .env，填入 ZHIPUAI_API_KEY=你的智谱Key

# 5. 首次入库（扫描 rag/ 下所有 .txt 文件，自动分块→向量化→写入 ChromaDB）
python ragmuti_doc_ingest.py

# 6. 启动 Web 应用（推荐）
streamlit run app.py

# 或者：终端版问答
python ragrag_pipeline.py
重要：所有命令必须在项目根目录执行。 内部依赖 sys.path 将根目录加入搜索路径，cd rag/ 再跑会 ModuleNotFoundError。
使用方式
启动 streamlit run app.py 后，在侧边栏选择三种模式：

模式	                  做什么	                              示例输入
智能对话	      Agent 自主决定是否调用工具	            "北京今天多少度？换算成华氏度？"
纯知识库问答	     仅检索本地文档库回答	                "缓解焦虑的方法有哪些？"
双 Agent 协作	  Researcher 搜集 + Writer出报告	       "介绍一下量子计算"
核心设计思路
1. ReAct 工具调用循环
ChatAgent 内部实现 react(user_input, max_steps=5)：
模型思考 → 决定调用哪个工具 → 执行并拿到结果 → 再思考 → 直到不再需要工具或达上限
工具调用走 call_tool_with_retry：失败自动重试 2 次 + 指数退避，最终仍失败则 degraded_message 降级回答（不抛异常、不崩）
2. 双 Agent 角色契约
Researcher：只搜集、只汇报原始素材、不写文章、不回答用户
Writer：基于素材重写为四段式文章（标题/核心概念/应用场景/未来展望）
分离"事实搜集"和"内容生成"两个角色，抑制模型幻觉。
3. RAG 入库与检索一致性
入库与检索必须保持三个值全局统一：
向量模型：embedding-3
向量维度：1024
Collection 名：rag_collection
4. Streamlit 状态管理
@st.cache_resource 缓存 Agent 实例 → 跨重跑保持对话记忆
st.session_state["ingested"] 去重闸门 → 避免文件重复入库阻塞渲染
st.session_state["messages"] 对话历史 → 手动重绘保证重跑后历史不丢
配置说明
config/config.yaml 关键配置项：
llm:
  name: "glm-4-flash"
  temperature: 0.3

embedding:
  name: "embedding-3"

retrieval:
  chroma_path: "./chorma_db"
  top_k: 3

system_prompt: |
  你是一个温柔可爱的 AI 助手...
  工具返回的数据就是权威来源，拿到就别反复确认。

writer_prompt: |
  你是一位资深科技专栏作家，请按以下结构撰写：
  1. 标题  2. 核心概念  3. 应用场景  4. 未来展望

logging:
  level: "INFO"
License
MIT
联系方式     2499257919@qq.com
如有问题或建议，欢迎提 Issue 或联系作者。