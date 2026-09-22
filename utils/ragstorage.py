import os
import chromadb
from chromadb.config import Settings
import streamlit as st

COLLECTION_NAME = "rag_collection"

@st.cache_resource
def get_chroma():
    """
    Streamlit 官方推荐的跨 session 单例方式。
    所有 worker 共享同一个 ChromaDB client 实例，
    数据不会出现"这个 worker 有、那个 worker 没有"的情况。
    """
    return chromadb.Client(
        settings=Settings(
            anonymized_telemetry=False,
            # 强制使用内存模式，避免磁盘路径问题
            is_persistent=False,
        )
    )

def get_collection():
    return get_chroma().get_or_create_collection(COLLECTION_NAME)