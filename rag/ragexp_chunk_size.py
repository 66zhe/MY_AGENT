import os
import sys
# 将项目根目录加入搜索路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chunker import smart_chunker
from rag.rag_embedding import RAGRetriever, get_embedding
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger("ExpChunkSize")

# 读取测试文档（和你入库时用的是同一份）
current_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(current_dir, "sample.txt")
with open(sample_path, "r", encoding="utf-8") as f:
    long_text = f.read()

logger.info(f"原文总长度: {len(long_text)} 字符")

# 要对比的 chunk_size 列表
chunk_sizes = [200, 400, 1000]
overlap = 80  # 固定 overlap，只变 chunk_size

test_query = "这篇文章讲了什么？"  # 换成你文档相关的具体问题

# 先用 RAGRetriever 拿到当前配置（chunk_size=400, overlap=80）的检索结果做基准
try:
    retriever = RAGRetriever()
    baseline_results = retriever.search(test_query, top_k=3, use_rewrite=True)
    logger.info("=== 基准检索结果 (chunk_size=400, overlap=80) ===")
    for item in baseline_results:
        logger.info(f"  Top{item['rank']} 相似度={item['score']:.4f} | 来源={item['source']}")
        logger.info(f"  内容: {item['content'][:150]}...")
except Exception as e:
    logger.error(f"初始化 RAGRetriever 失败: {e}")
    baseline_results = []

# 对每个 chunk_size 重新分块并打印统计
print("\n" + "=" * 60)
print("【Chunk Size 对比实验】")
print("=" * 60)

for cs in chunk_sizes:
    chunks = smart_chunker(long_text, chunk_size=cs, overlap=overlap)
    print(f"\n--- chunk_size={cs}, overlap={overlap} ---")
    print(f"  分块数: {len(chunks)}")
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"  平均每块长度: {avg_len:.1f}")
    print(f"  第一块: {chunks[0][:80]}...")
    if len(chunks) > 1:
        print(f"  第二块: {chunks[1][:80]}...")
    
    # 检查是否有块被硬切（长度接近 chunk_size）
    full_chunks = [c for c in chunks if len(c) >= cs * 0.9]
    print(f"  接近满长度的块数: {len(full_chunks)}/{len(chunks)}")
    print()

logger.info("Chunk Size 对比实验完成")