import os
import sys

# 将项目根目录加入搜索路径（解决 from utils 找不到的问题）
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.config import Settings
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from utils.logger import get_logger
from chunker import smart_chunker
from utils.ragstorage import get_chroma

# 初始化 logger
logger = get_logger("RegVectorDB")

# 加载环境变量
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)


def get_embedding(text):
    """获取智谱 embedding-3 向量（1024维）"""
    response = client.embeddings.create(
        model="embedding-3",
        input=[text],
        dimensions=1024  # 锁死 1024 维
    )
    return response.data[0].embedding


def main():
    # 确定项目根目录（regvetor_db.py 的上一级）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)  # D:\my_agent

    # ChromaDB 路径（与 search.py 保持一致）
    db_path = os.path.join(root_dir, "chorma_db")
    logger.info(f"正在初始化 ChromaDB 数据库: {db_path}")

    chroma_client = get_chroma()

    collection_name = "rag_collection"
    # 如果已存在则删除重建（确保数据干净）
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    logger.info(f"创建/重置集合: {collection_name}")

    # 读取文本
    sample_path = os.path.join(current_dir, "sample.txt")
    with open(sample_path, "r", encoding="utf-8") as f:
        long_text = f.read()

    logger.info(f"正在读取文件: {sample_path}，原文长度 {len(long_text)} 字符")

    # 语义切分
    chunks = smart_chunker(long_text, chunk_size=400, overlap=80)
    logger.info(f"切块完成，共 {len(chunks)} 个片段。开始向量化入库...")

    # 批量入库
    for i, chunk in enumerate(chunks):
        logger.info(f"  [{i+1}/{len(chunks)}] 正在向量化第 {i+1} 块...")
        vec = get_embedding(chunk)
        collection.add(
            embeddings=[vec],
            documents=[chunk],
            ids=[f"chunk_{i}"],
            metadatas=[{"source": "sample.txt", "index": i}]
        )

    logger.info(f"数据入库成功！共 {len(chunks)} 条 -> {collection_name}")


if __name__ == "__main__":
    main()