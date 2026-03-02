from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import config
from app.logger.logger import get_logger
import uuid

logger = get_logger(__name__)

def chunk_document(document: dict) -> list[dict]:
    """
    Split a document's text into overlapping chunks.
    
    Args:
        document: Output from load_document() 
                  {"text": "...", "metadata": {...}}
    
    Returns:
        List of chunks, each with text + metadata
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,       # 1000 chars per chunk
        chunk_overlap=config.CHUNK_OVERLAP, # 150 chars overlap
        separators=["\n\n", "\n", ". ", " ", ""] 
    )
    
    raw_chunks = splitter.split_text(document["text"])
    
    chunks = []
    for i, chunk_text in enumerate(raw_chunks):
        chunk = {
            "chunk_id": str(uuid.uuid4()),
            "doc_id": document["metadata"]["source"],
            "chunk_index": i,
            "text": chunk_text,
            "metadata": {
                "source": document["metadata"]["source"],
                "file_type": document["metadata"]["file_type"],
                "chunk_index": i,
                "total_chunks": len(raw_chunks)
            }
        }
        chunks.append(chunk)
    
    logger.info(
        f"Chunked: {document['metadata']['source']} → "
        f"{len(chunks)} chunks"
    )
    
    return chunks