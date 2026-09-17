import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.config import Settings
from zhipuai import ZhipuAI
from dotenv import load_dotenv
from utils.logger import get_logger
from rank_bm25 import BM25Okapi
import re
import math

logger = get_logger("HybridSearch")

load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)


def get_embedding(text):
    """获取智谱 embedding-3 向量 (1024维) """
    response = client.embeddings.create(
        model="embedding-3",
        input=[text],
        dimensions=1024
    )
    return response.data[0].embedding


def chinese_tokenize(text):
    """简单的中文分词: 逐字切分"""
    chars = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', text)
    return chars 


class HybridRetriver:
    def __init__(self, db_path=None, collection_name="rag_collection"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        self.db_path = os.path.join(root_dir, "chorma_db")

        self.chroma_client = chromadb.PersistentClient(
            path=self.db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection(name=collection_name)

        all_docs = self.collection.get()["documents"]
        self.documents = all_docs
        self.doc_ids = self.collection.get()["ids"]

        self.bm25 = BM25Okapi([chinese_tokenize(doc) for doc in all_docs])

        logger.info(f"HybridRetriever 初始化完成, 共 {len(all_docs)} 条文档")

    def search_hybrid(self, query, top_k=3, alpha=0.5):
        """
        混合检索: BM25 关键词检索 + 向量语义检索, 加权融合

        :param query: 用户查询
        :param top_k: 返回前 k 个结果
        :param alpha: 向量检索权重 (0=纯BM25, 1=纯向量, 0.5=各一半)
        :return: 融合后的排序结果列表
        """
        query_tokens = chinese_tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)

        # BM25 分数归一化
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        bm25_norm = [s / max_bm25 for s in bm25_scores]

        # 向量检索
        query_vector = get_embedding(query)
        if query_vector:
            vector_results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=max(top_k * 2, 5)
            )

            vector_scores = {doc_id: 0.0 for doc_id in self.doc_ids}
            for i, doc_id in enumerate(vector_results["ids"][0]):
                sim = 1 - vector_results["distances"][0][i]
                vector_scores[doc_id] = sim  # ✅ 修复：原来写成了 vector_results[doc_id]
        else:
            vector_scores = {doc_id: 0.0 for doc_id in self.doc_ids}

        # 加权融合
        combined_scores = []
        for i, doc_id in enumerate(self.doc_ids):
            bm25_s = bm25_norm[i]
            vector_s = vector_scores.get(doc_id, 0.0)
            combined = alpha * vector_s + (1 - alpha) * bm25_s
            combined_scores.append({
                "id": doc_id,
                "doc": self.documents[i],
                "bm25_score": bm25_s,
                "vector_score": vector_s,       # ✅ 修复：原来写成了 combined
                "combined_score": combined      # ✅ 修复：新增独立 key
            })

        combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        return combined_scores[:top_k]


if __name__ == "__main__":
    try:
        retriever = HybridRetriver()  # ✅ 现在不会报错了
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)

    test_query = "这篇文章讲了什么？"
    print(f"\n查询: {test_query}")
    print("=" * 50)

    results = retriever.search_hybrid(test_query, top_k=3, alpha=0.5)
    for i, item in enumerate(results):
        print(f"\nTop{i+1} (综合分数={item['combined_score']:.4f})")
        print(f" BM25分数={item['bm25_score']:.4f} | 向量分数={item['vector_score']:.4f}")
        print(f" 内容: {item['doc'][:200]}...")