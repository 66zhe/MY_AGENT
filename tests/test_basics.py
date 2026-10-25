"""
D3 主线 B · 单元测试（pytest）
仅 import 纯函数，不触碰 app.py 顶层 Streamlit / ChromaDB 连接。
运行：cd D:/my_agent && D:/ANACONDA-3/python.exe -m pytest tests/ -q
"""
import os
import sys

# 加固：确保项目根目录在 sys.path（python -m pytest 运行时 cwd 即根目录）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rag.chunker import smart_chunker


def dedup_preserve_order(seq):
    """保序去重：dict.fromkeys 保持首次出现顺序并去重。"""
    return list(dict.fromkeys(seq))


def test_smart_chunker_empty():
    """空串 / 纯空白串都应返回 []"""
    assert smart_chunker("") == []
    assert smart_chunker("   \n  ") == []


def test_smart_chunker_splits_long_text():
    """长文本应被切成多块，且每块非空、长度不超过 chunk_size+overlap 上限"""
    text = "人工智能是计算机科学的一个分支。" * 50  # 约 650+ 字符
    chunks = smart_chunker(text, chunk_size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)
    assert all(len(c) <= 200 + 40 for c in chunks)


def test_dedup_preserves_order():
    """保序去重：顺序保持、重复剔除"""
    assert dedup_preserve_order([1, 2, 2, 1, 3, 4, 6]) == [1, 2, 3, 4, 6]
    assert dedup_preserve_order(["a", "b", "a", "c"]) == ["a", "b", "c"]
