# MY_AGENT - 基于智谱 GLM 的 RAG 知识库问答助手

## 项目简介

一个基于智谱 GLM 大模型的 RAG（检索增强生成）知识库问答系统。支持将多篇文档入库，针对文档内容提问，系统通过语义检索匹配相关段落，并调用大模型基于资料生成回答，有效减少幻觉。

项目包含两大核心模块：
- **RAG 知识库问答**：上传文档 → 智能分块 → 向量化 → 语义检索 + LLM 生成回答（Web 界面）
- **多智能体协作系统**：Researcher Agent（工具调用查资料）+ Writer Agent（撰写文章），基于 Function Calling 实现双 Agent 协作

## 功能特性

### RAG 知识库（Day 31-42）

- 语义感知文本分块（smart_chunker，按段落→换行→句号→逗号逐级回退切分，支持 overlap）
- 文本向量化（智谱 embedding-3，1024 维）
- 向量检索（ChromaDB 本地库，余弦相似度 Top-K）
- 查询重写（rewrite_query，让大模型将自然语言问题改写为更适合检索的陈述句）
- 多文档批量入库（自动扫描 rag/ 目录下所有 .txt 文件并入库）
- BM25 + 向量混合检索（加权融合，rank_bm25）
- RAG 全流程串联（检索 → 生成 → 回答，rag_pipeline.py）
- Streamlit Web 界面（文档上传、对话交互、知识库管理、使用示例）
- Streamlit Cloud 部署支持（.streamlit/ 配置 + Secrets 管理）

### 多智能体协作（Day 22-30）

- Function Calling 工具调用（ChatAgent 核心类，自动注册工具 → 模型决定调用 → 执行 → 返回结果）
- 双 Agent 协作流水线（Researcher 查资料 + Writer 写文章）
- 模拟工具：天气查询、全网搜索（可替换为真实 API）

## 技术栈

- 大模型：智谱 glm-4-flash
- Embedding：智谱 embedding-3（1024 维）
- 向量数据库：ChromaDB（本地持久化，路径 ./chorma_db）
- 关键词检索：rank_bm25
- Web 框架：Streamlit 1.64.0
- 开发语言：Python 3

## 目录结构
MY_AGENT/
├── app.py                  # Streamlit Web 界面（Day 42：美化 + 上传 + 对话）
├── main.py                 # 双 Agent 协作主入口（Researcher + Writer）
├── requirements.txt        # 项目依赖
├── .env                    # 环境变量（ZHIPUAI_API_KEY，不要上传 GitHub）
├── .env.example            # 环境变量模板
├── README.md               # 项目说明
├── .streamlit/             # Streamlit Cloud 部署配置
│   └── config.toml         # 服务器 + 主题配置
├── agent/                  # 多智能体核心模块
│   ├── chat_agent.py       # ChatAgent 核心类（Function Calling）
│   └── writer_agent.py     # WriterAgent（文章撰写）
├── Config/                 # 全局配置
│   └── config.yaml         # 模型、embedding、检索、Prompt 配置
├── Tools/                  # 模拟工具
│   ├── weather_tool.py     # 天气查询工具
│   └── search_tool.py      # 搜索工具
├── Utils/                  # 通用工具
│   └── logger.py           # 日志工具（控制台 + 文件输出）
├── rag/                    # RAG 核心模块
│   ├── chunker.py          # 智能文本分块器
│   ├── rag_embedding.py    # 向量化 + 查询重写 + RAGRetriever
│   ├── rag_generator.py    # RAG 生成回答（LLM 基于检索结果生成）
│   ├── rag_pipeline.py     # 完整 RAG 问答链路
│   ├── regvetor_db.py      # 单文档入库脚本
│   ├── muti_doc_ingest.py  # 多文档批量入库
│   ├── search.py           # 向量检索（基础版）
│   ├── raghybrid_search.py # BM25 + 向量混合检索（实验）
│   ├── ragexp_chunk_size.py # chunk_size 调参实验
│   ├── ragexp_overlap.py   # overlap 调参实验
│   ├── sample.txt          # 测试文档
│   ├── article1.txt         # 测试文档
│   ├── article2.txt         # 测试文档
│   └── article3.txt         # 测试文档
└── utils/                  # RAG 辅助工具
    └── ragstorage.py       # ChromaDB 客户端单例
## 快速开始

### 1. 克隆仓库

git clone 你的仓库地址
cd MY_AGENT

### 2. 创建并激活虚拟环境

python -m venv venv
venvScriptsactivate     # Windows
source venv/bin/activate  # Mac/Linux

### 3. 安装依赖

pip install -r requirements.txt

### 4. 配置 API Key

复制 `.env.example` 为 `.env`，填入你自己的智谱 API Key：

ZHIPUAI_API_KEY=你的key

### 5. 文档入库（首次运行需执行）

python ragmuti_doc_ingest.py

入库脚本会自动扫描 `rag/` 目录下所有 `.txt` 文件，分块、向量化后存入 ChromaDB。如需新增文档，将 `.txt` 文件放入 `rag/` 目录后重新运行此命令即可。

### 6. 启动问答

**方式一：终端交互问答**

python ragrag_pipeline.py

在终端输入问题，输入 `quit` 退出。

**方式二：Streamlit Web 界面（推荐）**

streamlit run app.py

浏览器打开本地链接，即可通过网页上传文档、查看知识库统计、清空知识库、进行对话问答。

**方式三：多智能体协作**

python main.py

进入双 Agent 协作模式，输入主题后 Researcher 自动调用工具查资料，Writer 撰写文章。

### 7. 部署到 Streamlit Cloud

1. 确保代码已推送到 GitHub（`.env` 不要上传）
2. 在 `.streamlit/` 目录下已有 `config.toml` 配置
3. 前往 Streamlit Cloud 用 GitHub 账号登录
4. 点击 "New app"，选择你的 MY_AGENT 仓库，分支选 main
5. 在 Settings → Secrets 中填入 `ZHIPUAI_API_KEY`
6. 点击 Deploy，即可获得公网访问链接

## 说明

- `chorma_db/` 为本地向量数据库，不纳入版本管理，clone 后运行入库脚本会自动重建。
- `rag/` 下 `ragexp_` 开头的脚本为调参实验脚本（chunk_size 对比、overlap 对比），`raghybrid_search.py` 为 BM25+向量混合检索实验，均为非核心功能，保留在原位方便参考。
- `agent`、`main.py` 为多智能体协作模块（Day 22-30 阶段），与 RAG 知识库可独立运行。
- 所有脚本内部通过 `sys.path` 把项目根目录加入了模块搜索路径，请统一在项目根目录下运行，不要 cd 进子目录再运行。
- Streamlit Cloud 部署时，`/tmp` 目录为唯一可写路径，ChromaDB 数据会保存在 `/tmp/chroma_db`，重启后需重新上传文档。
- 已知问题修复：`rag_embedding.py` 的 `search` 方法中，每次查询前需刷新 collection 引用（`get_or_create_collection`），避免清空知识库后出现 `NotFoundError`。

## 项目阶段
阶段   时间范围   内容
Day 22-30   多智能体   Function Calling、双 Agent 协作（Researcher + Writer）、工具注册
Day 31-35   RAG 基础   文本分块、向量化、ChromaDB、语义检索、RAG 全流程串联
Day 36-41   RAG 进阶   多文档入库、查询重写、效果调参、GitHub 整理、Streamlit 入门
Day 42   上线部署   界面美化、侧边栏管理、Web 部署到 Streamlit Cloud