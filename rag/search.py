import os
import sys

# 将项目根目录加入搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from utils.logger import get_logger

# 初始化 logger
logger = get_logger("Retriever")

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


class Retriever:
    def __init__(self, db_path, collection_name="rag_collection"):
        logger.info(f"正在连接 ChromaDB: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.count = self.collection.count()
        logger.info(f"Retriever 初始化完成，集合: {collection_name}，共 {self.count} 条")

    def search(self, query, top_k=3):
        """语义检索：返回最相关的 top_k 个文本块"""
        query_vec = get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        # 距离转相似度（ChromaDB 默认用 L2 距离）
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        return list(zip(documents, metadatas, distances))


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    db_path = os.path.join(root_dir, "chorma_db")

    retriever = Retriever(db_path)

    print("\n" + "=" * 60)
    print("💬 欢迎使用 RAG 检索系统")
    print("=" * 60)

    while True:
        try:
            query = input("\n🔍 请输入查询 (quit 退出): ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                print("👋 再见！")
                break
            if not query:
                continue

            results = retriever.search(query, top_k=3)

            print(f"\n📋 查询：{query}")
            for i, (doc, meta, dist) in enumerate(results):
                # L2 距离转相似度（经验公式）
                similarity = 1.0 / (1.0 + dist)
                print(f"  Top{i+1} 相似度 = {similarity:.4f}  来源 = {meta.get('source', 'unknown')}")
                print(f"    内容预览: {doc[:100]}...")

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break