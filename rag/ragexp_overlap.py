import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from chunker import smart_chunker
from dotenv import load_dotenv
from utils.logger import get_logger

logger = get_logger("ExpOverlap")

current_dir = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(current_dir, "sample.txt")
with open(sample_path, "r", encoding="utf-8") as f:
    long_text = f.read()

# 固定 chunk_size=400，变 overlap
chunk_size = 400
overlaps = [0, 50, 80, 100]

print("\n" + "=" * 60)
print("【Overlap 对比实验】")
print("=" * 60)

for ov in overlaps:
    chunks = smart_chunker(long_text, chunk_size=chunk_size, overlap=ov)
    print(f"\n--- chunk_size={chunk_size}, overlap={ov} ---")
    print(f"  分块数: {len(chunks)}")
    avg_len = sum(len(c) for c in chunks) / len(chunks)
    print(f"  平均每块长度: {avg_len:.1f}")
    
    # 检查重叠情况
    if ov > 0 and len(chunks) > 1:
        # 取第二块的前 ov 字符和第一块的末尾 ov 字符对比
        prev_tail = chunks[0][-ov:] if len(chunks[0]) > ov else chunks[0]
        curr_head = chunks[1][:ov]
        match = sum(1 for a, b in zip(prev_tail, curr_head) if a == b)
        print(f"  重叠验证: 第一块末尾[:{ov}]与第二块头部前{match}/{ov}字符匹配")
    
    # 看有没有语义截断（块结尾突然中断在一个词中间）
    last_chunk = chunks[-1]
    if len(last_chunk) > 20:
        # 检查最后几个字符是否是完整句子的结尾
        ends_with_punct = last_chunk.rstrip()[-1] in '。！？）"》'
        print(f"  最后一块结尾字符: '{last_chunk.rstrip()[-1]}' {'✓ 标点结尾' if ends_with_punct else '✗ 可能被截断'}")
    print()

logger.info("Overlap 对比实验完成")