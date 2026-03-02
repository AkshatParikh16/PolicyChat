from pinecone import Pinecone
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)

# Singleton Pinecone client
pc = None
index = None

def get_pinecone_index():
    """Get or create Pinecone index connection (singleton)."""
    global pc, index
    
    if index is None:
        logger.info("Connecting to Pinecone...")
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        index = pc.Index(config.PINECONE_INDEX_NAME)
        logger.info(f"Connected to Pinecone index: {config.PINECONE_INDEX_NAME}")
    
    return index


def upsert_chunks(chunks: list[dict]) -> bool:
    """
    Store embedded chunks in Pinecone.
    
    Args:
        chunks: Chunks with embeddings from embed_chunks()
    
    Returns:
        True if successful
    """
    try:
        pinecone_index = get_pinecone_index()
        
        vectors = []
        for chunk in chunks:
            vectors.append({
                "id": chunk["chunk_id"],
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "source": chunk["metadata"]["source"],
                    "file_type": chunk["metadata"]["file_type"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"],
                    "doc_id": chunk["doc_id"]
                }
            })
        
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            pinecone_index.upsert(vectors=batch)
            logger.info(f"Upserted batch {i//batch_size + 1}: {len(batch)} vectors")
        
        logger.info(f"Total upserted: {len(vectors)} vectors")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone upsert failed: {e}")
        raise


def search_pinecone(query_embedding: list[float],
                    top_k: int = None,
                    doc_id: str = None) -> list[dict]:
    """
    Search Pinecone for similar chunks.
    
    Args:
        query_embedding: Embedded query vector
        top_k: Number of results to return
        doc_id: Optional filter to search only specific document
    
    Returns:
        List of matching chunks with scores
    """
    try:
        pinecone_index = get_pinecone_index()
        top_k = top_k or config.TOP_K
        
        filter_dict = {"doc_id": {"$eq": doc_id}} if doc_id else None
        
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        chunks = []
        for match in results.matches:
            chunks.append({
                "chunk_id": match.id,
                "text": match.metadata.get("text", ""),
                "score": match.score,
                "metadata": {
                    "source": match.metadata.get("source", ""),
                    "file_type": match.metadata.get("file_type", ""),
                    "chunk_index": match.metadata.get("chunk_index", 0),
                    "doc_id": match.metadata.get("doc_id", "")
                }
            })
        
        logger.info(f"Pinecone search returned {len(chunks)} results")
        return chunks
        
    except Exception as e:
        logger.error(f"Pinecone search failed: {e}")
        raise


def delete_document(doc_id: str) -> bool:
    """
    Delete all chunks belonging to a document.
    Useful when re-uploading an updated policy.
    """
    try:
        pinecone_index = get_pinecone_index()
        
        pinecone_index.delete(
            filter={"doc_id": {"$eq": doc_id}}
        )
        
        logger.info(f"Deleted all chunks for doc_id: {doc_id}")
        return True
        
    except Exception as e:
        logger.error(f"Pinecone delete failed: {e}")
        raise