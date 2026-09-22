import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.config import Settings
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from utils.logger import get_logger
from chunker import smart_chunker
from utils.ragstorage import get_chroma

logger = get_logger("MultiDoclngest")

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")

client = ZhipuAI(api_key=api_key)

def get_embedding(text):
    """获取智谱 embedding-3 向量 (1024维) """
    response = client.embeddings.create(
        model="embedding-3",
        input=[text],
        dimensions=1024
    )
    return response.data[0].embedding

def load_documents(folder_path):
    """
    读取指定文件夹中所有 .txt 文件, 返回文件路径列表。
    :param folder_path: 文件夹路径
    :return: 所有 .txt 文件的绝对路径列表
    """

    txt_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".txt")
    ]
    logger.info(f"在 {folder_path} 中找到 {len(txt_files)} 个 .txt 文件")
    for f in txt_files:
        logger.info(f" - {os.path.basename(f)}")
    return txt_files

def ingest_documents(collection, txt_files):
    """
    对每个 .txt 文件执行 : 读取 → 分块 → 向量化 → 存入 ChromaDB
    :param collection: ChromaDB collection 对象
    :param txt_files: 文件路径列表
    """
    total_chunks = 0
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        logger.info(f"\n正在处理文件: {file_name}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info(f" 文件 {file_name} 读取完成, 长度: {len(text)} 字符")

        chunks = smart_chunker(text, chunk_size=400, overlap=80)
        logger.info(f" 切块完成, 共 {len(chunks)} 个片段")

        for i, chunk in enumerate(chunks):
            vec = get_embedding(chunk)
            collection.add(
                embeddings=[vec],
                documents=[chunk],
                ids=[f"{file_name}_chunk_{i}"],
                metadatas=[{"source": file_name, "index":i}]
            )
            logger.debug(f" [{file_name}] 第 {i+1}/{len(chunks)} 块已入库")

        total_chunks += len(chunks)

    logger.info(f"\n全部处理完成! 共处理 {len(txt_files)} 个文件, {total_chunks} 个文本块")

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rag_dir = current_dir

    root_dir = os.path.dirname(current_dir)
    db_path = os.path.join(root_dir, "chorma_db")
    logger.info(f"正在初始化 ChromaDB 数据库: {db_path}")

    chroma_client = get_chroma()
    collection_name = "rag_collection"

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=collection_name)
    logger.info(f"创建/重置集合: {collection_name}")

    txt_files = load_documents(rag_dir)
    if not txt_files:
        logger.warning(f"在 {rag_dir} 中未找到任何 .txt 文件, 请放入文章后重试")
        return

    ingest_documents(collection, txt_files)

    logger.info(f"集合 {collection_name} 当前共 {collection.count()} 条数据")


if __name__ == "__main__":
    main()