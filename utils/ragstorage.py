import os
import chromadb
from chromadb.config import Settings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(ROOT_DIR, "chorma_db")
COLLECTION_NAME = "rag_collection"

_client = None

def get_chroma():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=Settings(anonymized_telemetry=False)
            )
    return _client

def get_collection():
    return get_chroma().get_or_create_collection(COLLECTION_NAME)