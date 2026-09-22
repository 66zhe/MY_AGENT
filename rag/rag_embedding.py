import os
import sys

# --- 确保项目根目录在 sys.path 中，解决找不到 utils 的问题 ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from chromadb.config import Settings
from zhipuai import ZhipuAI
from dotenv import load_dotenv

from utils.logger import get_logger
from utils.ragstorage import get_chroma

# 获取 logger
logger = get_logger("Retriever")

# 加载环境变量（获取 API Key）
load_dotenv()
api_key = os.getenv("ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)


def get_embedding(text):
    """
    使用智谱 AI 获取文本的 Embedding 向量
    :param text: 需要向量化的文本
    :return: 1024维向量列表
    """
    try:
        response = client.embeddings.create(
            model="embedding-3",
            input=[text],
            dimensions=1024
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"获取向量失败: {e}")
        return None


def rewrite_query(query, top_k=3):
    """
    查询重写：让大模型把用户的自然语言问题改写为更适合语义检索的陈述句。
    :param query: 用户原始问题
    :param top_k: 返回几个改写版本
    :return: 改写后的查询字符串（多个用空格拼接）
    """
    try:
        messages = [
            {"role": "system", "content": (
                "你是一个查询重写专家。用户的提问往往是疑问句或口语化的，"
                "请将其改写为1-2句包含丰富关键词的陈述句，方便后续做语义检索。"
                "只输出改写后的句子，不要任何解释。"
            )},
            {"role": "user", "content": f"原始查询：{query}"}
        ]
        response = client.chat.completions.create(
            model="glm-4-flash",  # 轻量模型，速度快、成本低
            messages=messages,
            temperature=0.3,
            max_tokens=200
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(f"查询重写: '{query}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"查询重写失败，使用原始查询: {e}")
        return query  # 降级策略：重写失败则用原查询


class RAGRetriever:
    def __init__(self, collection_name="rag_collection", client=None):
        self.chroma_client = client or get_chroma()     
        self.collection_name = collection_name

        
        # 确保集合存在
        if self.collection_name not in [c.name for c in self.chroma_client.list_collections()]:
            logger.error(f"获取集合 {self.collection_name} 失败: Collection does not exist")
            raise ValueError(f"集合 {self.collection_name} 不存在，请先运行 regvetor_db.py 入库")

        self.collection = self.chroma_client.get_collection(name=self.collection_name)
        logger.info(f"Retriever 初始化完成, 集合: {self.collection_name}, 共 {self.collection.count()} 条")

    def search(self, query, top_k=3, use_rewrite=True):
        """
        语义检索
        :param query: 用户查询
        :param top_k: 返回 Top K 结果
        :param use_rewrite: 是否启用查询重写（默认开启）
        :return: 匹配结果列表
        """
        # 查询重写
        if use_rewrite:
            search_query = rewrite_query(query)
        else:
            search_query = query

        # 获取查询向量
        query_vector = get_embedding(search_query)
        if not query_vector:
            logger.error("无法获取查询向量")
            return []

        # 执行相似度搜索
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        # 组装返回
        output = []
        for i in range(len(documents)):
            output.append({
                "rank": i + 1,
                "score": 1 - distances[i],  # 转为相似度分数
                "content": documents[i],
                "source": metadatas[i].get("source", "unknown"),
            })
        return output


if __name__ == "__main__":
    print("=" * 50)
    print("💬 欢迎使用 RAG 检索系统（含查询重写）")
    print("=" * 50)

    try:
        retriever = RAGRetriever()
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        sys.exit(1)

    while True:
        try:
            query = input("\n🔎 请输入查询（quit 退出）：").strip()
            if not query:
                continue
            if query.lower() in ("quit", "exit", "退出"):
                print("👋 再见！")
                break

            print(f"\n📄 查询：{query}")
            results = retriever.search(query, top_k=3, use_rewrite=True)

            for item in results:
                print(f"  Top{item['rank']} 相似度 = {item['score']:.4f} | 来源 = {item['source']}")
                print(f"  内容：{item['content'][:200]}...")
                print()

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break