import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.logger import get_logger

from .rag_embedding import RAGRetriever
from .rag_generator import generate_answer

logger = get_logger("RAGPipeline")

# 全局初始化一个检索器, 程序启动时连一次库即可, 避免每次提问都重连
retriver = RAGRetriever()

def rag_answer(question: str) -> str:
    """
    RAG 全流程主函数: 检索 -> 生成 -> 返回回答。
    :param question: 用户的问题
    :return: 大模型基于检索到的资料生成的回答
    """

    chunks = retriver.search(question, top_k=3, use_rewrite=True)

    print("\n 检索到的参考资料: ")
    for item in chunks:
        print(f" Top{item['rank']} 相似度={item['score']:.4f} | 来源={item['source']}")
        print(f"  {item['content'] [:80]}...")

    answer = generate_answer(question, chunks)
    return answer

def main():
    print("=" * 60)
    print("🤖 RAG 全流程问答系统已就绪 (Day35: 检索 + 生成 已打通)")
    print("💡 提示: 请输入与 sample.txt 相关的问题, 输入 quit 退出")
    print("=" * 60)

    while True:
        try:
            question = input("\n 👤 请输入问题: ").strip()
            if not question:
                continue
            if question.lower() in ["quit", "exit", "退出", "q"]:
                print("👋 再见! ")
                break

            answer = rag_answer(question)
            print(f"\n 回答: \n{answer}")

        except KeyboardInterrupt:
            print("\n👋 再见! ")
            break
        except Exception as e:
            logger.error(f"运行出错: {e}")
            print(f"出错了: {e}")

if __name__ == "__main__":
    main()