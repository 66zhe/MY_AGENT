import os
import re


def smart_chunker(text, chunk_size=400, overlap=80):
    """
    语义感知文本分块（纯 Python 实现）。
    策略：段落 → 换行 → 句号 → 感叹/问号 → 逗号 → 硬切，逐级回退。
    :param text: 输入的长文本
    :param chunk_size: 每块目标字符数
    :param overlap: 相邻块重叠字符数
    :return: 分块后的文本列表
    """
    if not text or not text.strip():
        return []
    separators = [
        ("\n\n", lambda t: re.split(r'\n\s*\n', t)),
        ("\n",   lambda t: t.split('\n')),
        ("。",   lambda t: t.split('。')),
        ("！",   lambda t: t.split('！')),
        ("？",   lambda t: t.split('？')),
        ("；",   lambda t: t.split('；')),
        ("，",   lambda t: t.split('，')),
    ]

    def _chunk_segment(segment, size, sep_idx=0):
        """递归切分单个段落/片段"""
        if len(segment) <= size:
            return [segment]

        if sep_idx >= len(separators):
            # 所有分隔符都用完了，按字符硬切
            return [segment[i:i+size] for i in range(0, len(segment), size - overlap)]

        sep, split_fn = separators[sep_idx]
        parts = split_fn(segment)

        # 尝试用当前分隔符切分
        chunks = []
        current_chunk = ""

        for i, part in enumerate(parts):
            # 恢复分隔符（最后一个除外）
            suffix = sep if i < len(parts) - 1 and part else ""
            candidate = current_chunk + part + suffix

            if len(candidate) > size and current_chunk:
                # 当前块已满，保存并开始新块
                chunks.append(current_chunk.strip())
                current_chunk = part + suffix
            else:
                current_chunk = candidate

        if current_chunk:
            chunks.append(current_chunk.strip())

        # 如果切完仍有超长块，递归用下一级分隔符继续切
        final_chunks = []
        for c in chunks:
            if len(c) > size:
                final_chunks.extend(_chunk_segment(c, size, sep_idx + 1))
            else:
                final_chunks.append(c)

        return final_chunks

    # 先按段落粗分，再逐段细切
    raw_paragraphs = re.split(r'\n\s*\n', text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    final_chunks = []
    for para in raw_paragraphs:
        final_chunks.extend(_chunk_segment(para, chunk_size))

    # 如果没有切出任何块（极短文本），直接返回原文
    if not final_chunks:
        return [text.strip()]

    # 处理 overlap：让相邻块之间有重叠
    if overlap > 0 and len(final_chunks) > 1:
        merged = []
        for i in range(len(final_chunks)):
            if i == 0:
                merged.append(final_chunks[i])
            else:
                # 取前一块的末尾 overlap 字符 + 当前块
                prev_tail = final_chunks[i - 1][-overlap:] if len(final_chunks[i - 1]) > overlap else final_chunks[i - 1]
                merged.append(prev_tail + final_chunks[i])
        return merged

    return final_chunks


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "sample.txt")

    with open(file_path, "r", encoding="utf-8") as f:
        long_text = f.read()

    result = smart_chunker(long_text, chunk_size=400, overlap=80)

    print(f"原文总长度: {len(long_text)}")
    print(f"一共分成了 {len(result)} 个块")
    print("=" * 50)

    for i, chunk in enumerate(result):
        print(f"--- 第 {i+1} 块 (长度: {len(chunk)}) ---")
        print(chunk[:100], "...")
        print("-" * 30)