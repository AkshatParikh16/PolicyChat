from rank_bm25 import BM25Okapi
from app.config import config
from app.logger.logger import get_logger
import pickle
import os
import re

logger = get_logger(__name__)

# In-memory BM25 index
bm25_index = None
stored_chunks = []

def tokenize(text: str) -> list[str]:
    """
    Simple tokenizer for BM25.
    Converts text to lowercase tokens.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()
    return tokens


def build_bm25_index(chunks: list[dict]) -> bool:
    """
    Build BM25 index from chunks.
    Called during document ingestion.
    """
    global bm25_index, stored_chunks
    
    try:
        # Add new chunks to stored chunks
        stored_chunks.extend(chunks)
        
        # Tokenize all chunk texts
        tokenized_chunks = [tokenize(chunk["text"]) for chunk in stored_chunks]
        
        # Build BM25 index
        bm25_index = BM25Okapi(tokenized_chunks)
        
        # Save to disk
        save_bm25_index()
        
        logger.info(f"BM25 index built with {len(stored_chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"BM25 index build failed: {e}")
        raise


def save_bm25_index() -> None:
    """Save BM25 index and chunks to disk."""
    try:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        with open(config.BM25_INDEX_PATH, "wb") as f:
            pickle.dump({
                "bm25": bm25_index,
                "chunks": stored_chunks
            }, f)
        logger.info(f"BM25 index saved to {config.BM25_INDEX_PATH}")
    except Exception as e:
        logger.error(f"Failed to save BM25 index: {e}")
        raise


def load_bm25_index() -> bool:
    """Load BM25 index from disk."""
    global bm25_index, stored_chunks
    
    try:
        if not os.path.exists(config.BM25_INDEX_PATH):
            logger.warning("No BM25 index found on disk")
            return False
            
        with open(config.BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
            bm25_index = data["bm25"]
            stored_chunks = data["chunks"]
            
        logger.info(f"BM25 index loaded: {len(stored_chunks)} chunks")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load BM25 index: {e}")
        raise


def search_bm25(query: str,
                top_k: int = None,
                doc_id: str = None) -> list[dict]:
    """
    Search BM25 index for keyword matches.
    
    Args:
        query: User's question (raw text, not embedded)
        top_k: Number of results
        doc_id: Optional filter by document
    
    Returns:
        List of matching chunks with scores
    """
    global bm25_index, stored_chunks
    
    try:
        # Load from disk if not in memory
        if bm25_index is None:
            loaded = load_bm25_index()
            if not loaded:
                logger.warning("BM25 index empty — returning no results")
                return []
        
        top_k = top_k or config.BM25_TOP_K
        
        # Tokenize query
        tokenized_query = tokenize(query)
        
        # Get BM25 scores for all chunks
        scores = bm25_index.get_scores(tokenized_query)
        
        # Filter by doc_id if provided
        if doc_id:
            filtered = [
                (i, score) for i, score in enumerate(scores)
                if stored_chunks[i].get("doc_id") == doc_id
            ]
        else:
            filtered = list(enumerate(scores))
        
        # Sort by score descending
        filtered.sort(key=lambda x: x[1], reverse=True)
        
        # Take top_k
        top_results = filtered[:top_k]
        
        # Format results
        results = []
        for idx, score in top_results:
            if score > 0:  # only return relevant results
                chunk = stored_chunks[idx]
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": float(score),
                    "metadata": chunk["metadata"]
                })
        
        logger.info(f"BM25 search returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"BM25 search failed: {e}")
        raise