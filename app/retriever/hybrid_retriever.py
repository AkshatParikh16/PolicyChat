from app.retriever.pinecone_retriever import search_pinecone
from app.retriever.bm25_retriever import search_bm25
from app.embedder.embedder import embed_query
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)


def normalize_scores(results: list[dict]) -> list[dict]:
    """
    Normalize scores to 0-1 range.
    Makes Pinecone and BM25 scores comparable.
    """
    if not results:
        return results
    
    scores = [r["score"] for r in results]
    min_score = min(scores)
    max_score = max(scores)
    
    # Avoid division by zero
    if max_score == min_score:
        for r in results:
            r["normalized_score"] = 1.0
        return results
    
    for r in results:
        r["normalized_score"] = (r["score"] - min_score) / (max_score - min_score)
    
    return results


def hybrid_search(query: str,
                  top_k: int = None,
                  doc_id: str = None,
                  pinecone_weight: float = 0.7,
                  bm25_weight: float = 0.3) -> list[dict]:
    """
    Combine Pinecone semantic search + BM25 keyword search.
    
    Args:
        query: User's question
        top_k: Number of final results
        doc_id: Optional document filter
        pinecone_weight: Weight for semantic scores (default 0.7)
        bm25_weight: Weight for keyword scores (default 0.3)
    
    Returns:
        Merged, ranked list of chunks with confidence signals
    """
    top_k = top_k or config.TOP_K
    
    # Step 1: Embed query for Pinecone
    logger.info(f"Hybrid search: '{query[:50]}...'")
    query_embedding = embed_query(query)
    
    # Step 2: Run both searches in parallel conceptually
    pinecone_results = search_pinecone(
        query_embedding=query_embedding,
        top_k=top_k * 2,  # get more than needed, will trim after merging
        doc_id=doc_id
    )
    
    bm25_results = search_bm25(
        query=query,
        top_k=top_k * 2,
        doc_id=doc_id
    )
    
    # Step 3: Normalize scores separately
    pinecone_results = normalize_scores(pinecone_results)
    bm25_results = normalize_scores(bm25_results)
    
    # Step 4: Merge results by chunk_id
    merged = {}
    
    for chunk in pinecone_results:
        chunk_id = chunk["chunk_id"]
        merged[chunk_id] = {
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "metadata": chunk["metadata"],
            "pinecone_score": chunk["normalized_score"],
            "bm25_score": 0.0,  # default if not in BM25 results
            "raw_pinecone_score": chunk["score"]
        }
    
    for chunk in bm25_results:
        chunk_id = chunk["chunk_id"]
        if chunk_id in merged:
            # chunk found in BOTH — update bm25 score
            merged[chunk_id]["bm25_score"] = chunk["normalized_score"]
        else:
            # chunk only in BM25 — add it
            merged[chunk_id] = {
                "chunk_id": chunk_id,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "pinecone_score": 0.0,
                "bm25_score": chunk["normalized_score"],
                "raw_pinecone_score": 0.0
            }
    
    # Step 5: Calculate hybrid score
    for chunk_id, chunk in merged.items():
        chunk["hybrid_score"] = (
            pinecone_weight * chunk["pinecone_score"] +
            bm25_weight * chunk["bm25_score"]
        )
    
    # Step 6: Sort by hybrid score
    ranked = sorted(
        merged.values(),
        key=lambda x: x["hybrid_score"],
        reverse=True
    )
    
    # Step 7: Take top_k
    final_results = ranked[:top_k]
    
    # Step 8: Add confidence signals
    top_score = final_results[0]["hybrid_score"] if final_results else 0
    
    for chunk in final_results:
        chunk["confidence_signals"] = {
            "top_score": top_score,
            "found_in_both": chunk["pinecone_score"] > 0 and chunk["bm25_score"] > 0,
            "semantic_strong": chunk["pinecone_score"] > 0.7,
            "keyword_strong": chunk["bm25_score"] > 0.7
        }
    
    logger.info(
        f"Hybrid search complete: {len(final_results)} results | "
        f"top score: {top_score:.3f}"
    )
    
    return final_results