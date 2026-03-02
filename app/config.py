import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # LLM
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3:8b-instruct-q4_0")

    # Pinecone
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "policychat")

    # Chunking
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200

    # Retrieval
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    TOP_K: int = 5
    BM25_TOP_K: int = 5
    CONFIDENCE_THRESHOLD: float = 0.35

    # Paths
    DATA_DIR: str = "data"
    BM25_INDEX_PATH: str = "data/bm25_index.pkl"
    DOCUMENTS_JSON_PATH: str = "data/documents.json"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    APP_ENV: str = os.getenv("APP_ENV", "development")

config = Config()