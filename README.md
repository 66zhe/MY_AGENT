# MY_AGENT - 基于智谱 GLM 的 RAG 知识库问答助手

## 项目简介
一个本地运行的 RAG（检索增强生成）知识库问答系统。支持把多篇文档入库，
针对文档内容提问，系统会检索相关段落并调用大模型基于资料作答，减少幻觉。

## 功能特性
- 语义感知文本分块（smart_chunker，按段落→句子逐级切分）
- 文本向量化（智谱 embedding-3，1024 维）
- 向量检索（ChromaDB 本地库，余弦相似度 Top-K）
- 多文档批量入库（支持 article1~3.txt 等多篇文档）
- BM25 + 向量混合检索（加权融合，rank_bm25）
- 交互式问答（rag_pipeline.py 终端连续对话）

## 技术栈
- 大模型：智谱 glm-4-flash
- Embedding：智谱 embedding-3（1024 维）
- 向量数据库：ChromaDB（本地持久化，路径 ./chorma_db）
- 关键词检索：rank_bm25
- 开发语言：Python 3

## 目录结构
rag/
├── chunker.py          # 文本分块（语义感知 smart_chunker）
├── rag_embedding.py    # 向量化 + 入库
├── muti_doc_ingest.py  # 多文档批量入库（入库入口）
├── search.py           # 向量检索
├── rag_generator.py    # 生成回答
├── rag_pipeline.py     # 完整问答链路（主问答入口）
├── ragexp_chunk_size.py   # 实验：chunk_size 调参（非核心）
├── ragexp_overlap.py      # 实验：overlap 调参（非核心）
├── raghybrid_search.py    # 实验：BM25+向量混合检索（非核心）
├── sample.txt          # 测试文档
└── article1~3.txt      # 多文档测试素材

## 快速开始

### 1. 克隆仓库
git clone 你的仓库地址
cd MY_AGENT

### 2. 创建并激活虚拟环境
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Mac/Linux

### 3. 安装依赖
pip install -r requirements.txt

### 4. 配置 API Key
复制 .env.example 为 .env，填入你自己的智谱 API Key：
ZHIPUAI_API_KEY=你的key

### 5. 文档入库（首次运行需执行）
python ragmuti_doc_ingest.py

### 6. 启动问答
python ragrag_pipeline.py
然后在终端输入问题，输入 quit 退出。

## 说明
- chorma_db/ 为本地向量库，不纳入版本管理，clone 后运行入库脚本会自动重建。
- rag/ 下 exp / hybrid 结尾的脚本（ragexp_chunk_size.py、ragexp_overlap.py、
  raghybrid_search.py）为 Day 37 调参 / 混合检索实验脚本，保留在原位、非核心功能，
  运行方式同样是 `python rag脚本名.py`，同样需在项目根目录执行。
- 所有脚本内部通过 sys.path 把项目根目录加入了模块搜索路径，
  请统一在项目根目录（MY_AGENT 下）运行，不要 cd 进 rag/ 再运行，否则 import 会失败。