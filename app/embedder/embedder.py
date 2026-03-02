from sentence_transformers import SentenceTransformer
from app.config import config
from app.logger.logger import get_logger
import numpy as np

logger = get_logger(__name__)

# Load model ONCE at module level — not inside functions
# This is critical for performance
model = None

def get_embedding_model() -> SentenceTransformer:
    """Load embedding model (singleton pattern)."""
    global model
    if model is None:
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        model = SentenceTransformer(config.EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return model


def embed_text(text: str) -> list[float]:
    """
    Convert a single text string into a vector.
    Used for embedding user queries at search time.
    """
    embedding_model = get_embedding_model()
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed a list of chunks.
    Used during document ingestion.
    
    Args:
        chunks: Output from chunk_document()
    
    Returns:
        Same chunks with 'embedding' field added
    """
    embedding_model = get_embedding_model()
    
    # Extract just the text from each chunk
    texts = [chunk["text"] for chunk in chunks]
    
    # Embed all texts in one batch — much faster than one by one
    logger.info(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True
    )
    
    # Add embedding back to each chunk
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    
    logger.info(f"Embedding complete: {len(chunks)} chunks embedded")
    return chunks


def embed_query(query: str) -> list[float]:
    """
    Embed a user query for similarity search.
    Identical to embed_text but semantically named differently
    to make code more readable.
    """
    return embed_text(query)