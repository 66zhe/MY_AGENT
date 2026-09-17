import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zhipuai import ZhipuAI
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger("Generator")

load_dotenv()
api_key =os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("未找到环境变量 ZHIPUAI_API_KEY")
client = ZhipuAI(api_key=api_key)

MODEL_NAME = "glm-4-flash"

def generate_answer(question: str, content_chunks: list) -> str:
    """
    RAG 的 G: 把检索到的文档块叫给大模型, 基于资料生成回答。
    :param question: 用户问题
    :param context_chunks: RAGRetriever.search() 返回的字典列表
    每项含 {"rank", "score", "context", "source"}
    :return:
    大模型基于参考资料生成的回答文本
    """

    if not content_chunks:
        return"资料中没有提到相关内容, 无法回答该问题。"
    context = "\n\n".join(
         f" 【资料 {i + 1}】 {item['content']}" for i, item in enumerate(content_chunks)
     )

    prompt = (
        f"以下是参考资料: \n{context}\n\n"
        f"请基于以上资料回答用户问题: {question}\n"
        f"1.只依据参考资料回答, 不要凭记忆编造。\n"
        f"2.如果资料中没有提到答案, 请直接回答“资料中没有提到”。"
    )

    messages = [
        {"role": "system", "context": "你是一个严谨的问答助手, 只根据用户提供的参考资料回答问题。"},
        {"role": "user", "content": prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        logger.info(f"生成回答完成: {answer[:50]}...")
        return answer
    except Exception as e:
        logger.error(f"生成回答失败: {e}")
        return f"生成回答失败: {str(e)}"

if __name__ == "__main__":
    from rag_embedding import RAGRetriever

    retriever = RAGRetriever()

    test_questions = [
        "这篇文章讲了什么主要内容？",
        "文章提到了哪些缓解焦虑的办法？",
        "文章的作者是谁, 什么时候发表的？",
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"❓️ 问题: {q}")

        chunks = retriever.search(q, top_k=3)
        for item in chunks:
            print(f" [检索] Top{item['rank']} 相似度={item['score']:.4f} 来源={item['source']}")

        answer = generate_answer(q, chunks)
        print(f"💬 回答: {answer}")

    print("\n" + "=" * 60)
    print("🎉 Day34 生成回答测试完成")