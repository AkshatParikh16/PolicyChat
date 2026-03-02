from app.parser.document_parser import load_document
from app.chunker.text_chunker import chunk_document
from app.embedder.embedder import embed_chunks, embed_query
from app.retriever.pinecone_retriever import upsert_chunks, delete_document
from app.retriever.bm25_retriever import build_bm25_index, load_bm25_index
from app.retriever.hybrid_retriever import hybrid_search
from app.safety.safety_checker import evaluate_query
from app.llm import get_llm
from app.config import config
from app.logger.logger import get_logger

logger = get_logger(__name__)


def build_prompt(query: str, chunks: list[dict], evaluation: dict) -> str:
    """Build the final prompt sent to LLM."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        context_parts.append(
            f"[Source {i+1}: {source}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)
    prefix = evaluation.get("safe_message", "") or ""

    prompt = f"""You are a precise insurance policy analyst. Your job is to answer questions using ONLY the document chunks provided below.

STRICT RULES:
1. Use ONLY the information explicitly stated in the CONTEXT below — nothing else.
2. If the specific information asked about is NOT explicitly stated in the context, respond ONLY with: "This information is not mentioned in the document."
3. Do NOT mention other topics from the context that were not asked about. If asked about room rent, only discuss room rent — not nursing facilities, deductibles, or anything else.
4. Do NOT infer, suggest, or extrapolate beyond what is literally written.
5. Do NOT say "however, the document does mention X" if X is unrelated to the question.
6. Be concise and direct. State the exact figure, clause, or condition if present.
7. Always state which source document your answer comes from.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    return prefix + prompt


def ingest_document(file_path: str) -> dict:
    """
    Full ingestion pipeline:
    PDF/DOCX/TXT → parse → chunk → embed → store
    """
    logger.info(f"Starting ingestion: {file_path}")

    try:
        # Step 1: Parse document
        logger.info("Step 1/4: Parsing document...")
        document = load_document(file_path)

        # Step 2: Chunk document
        logger.info("Step 2/4: Chunking document...")
        chunks = chunk_document(document)

        # Step 3: Embed chunks
        logger.info("Step 3/4: Embedding chunks...")
        embedded_chunks = embed_chunks(chunks)

        # Step 4: Store in Pinecone + BM25
        logger.info("Step 4/4: Storing in vector DB + BM25...")
        upsert_chunks(embedded_chunks)
        build_bm25_index(embedded_chunks)

        summary = {
            "status": "success",
            "file": file_path,
            "total_chunks": len(chunks),
            "source": document["metadata"]["source"],
            "file_type": document["metadata"]["file_type"]
        }

        logger.info(f"Ingestion complete: {len(chunks)} chunks stored")
        return summary

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise


def answer_query(query: str, doc_id: str = None) -> dict:
    """
    Full query pipeline:
    Question → retrieve → evaluate → generate → answer
    """
    logger.info(f"Processing query: '{query[:50]}...'")

    try:
        # Step 1: Load BM25 if not in memory
        load_bm25_index()

        # Step 2: Hybrid retrieval
        logger.info("Step 1/4: Retrieving relevant chunks...")
        retrieved_chunks = hybrid_search(
            query=query,
            top_k=config.TOP_K,
            doc_id=doc_id
        )

        # Step 3: Safety + confidence evaluation
        logger.info("Step 2/4: Evaluating query safety + confidence...")
        evaluation = evaluate_query(query, retrieved_chunks)

        # Step 4: Handle not found
        if evaluation["final_decision"] == "not_found":
            return {
                "answer": evaluation["safe_message"],
                "decision": "not_found",
                "chunks_used": [],
                "confidence": "low",
                "risk_level": evaluation["risk"]["risk_level"]
            }

        # Step 5: Build prompt + generate answer
        logger.info("Step 3/4: Generating answer...")
        llm = get_llm()

        if not llm.is_available():
            raise Exception(
                "Ollama is not running. "
                "Open a terminal and run: ollama serve"
            )

        prompt = build_prompt(query, retrieved_chunks, evaluation)
        answer = llm.generate(prompt)

        # Step 6: Build response
        logger.info("Step 4/4: Building response...")
        response = {
            "answer": answer,
            "decision": evaluation["final_decision"],
            "confidence": evaluation["confidence"]["confidence"],
            "risk_level": evaluation["risk"]["risk_level"],
            "chunks_used": [
                {
                    "source": c["metadata"]["source"],
                    "chunk_index": c["metadata"]["chunk_index"],
                    "hybrid_score": c.get("hybrid_score", 0),
                    "text_preview": c["text"][:150] + "..."
                }
                for c in retrieved_chunks
            ],
            "retrieval_stats": {
                "total_chunks_retrieved": len(retrieved_chunks),
                "top_score": retrieved_chunks[0].get("hybrid_score", 0) if retrieved_chunks else 0,
            }
        }

        logger.info(f"Query complete: decision={evaluation['final_decision']}")
        return response

    except Exception as e:
        logger.error(f"Query pipeline failed: {e}")
        raise


def compare_policies(query: str,
                     doc_id_1: str,
                     doc_id_2: str) -> dict:
    """
    Compare two policies on a specific query.
    Searches each document in complete isolation.
    """
    logger.info(f"Comparing policies: {doc_id_1} vs {doc_id_2}")

    try:
        load_bm25_index()
        llm = get_llm()

        if not llm.is_available():
            raise Exception("Ollama is not running. Run: ollama serve")

        def get_policy_answer(doc_id: str) -> dict:
            """Get answer from one specific policy."""
            chunks = hybrid_search(
                query=query,
                top_k=config.TOP_K,
                doc_id=doc_id
            )

            evaluation = evaluate_query(query, chunks)

            if not chunks or evaluation["final_decision"] == "not_found":
                return {
                    "answer": "This information is not mentioned in the document.",
                    "confidence": "low",
                    "chunks_used": []
                }

            prompt = build_prompt(query, chunks, evaluation)
            answer = llm.generate(prompt)

            return {
                "answer": answer,
                "confidence": evaluation["confidence"]["confidence"],
                "chunks_used": [
                    {
                        "source": c["metadata"]["source"],
                        "chunk_index": c["metadata"]["chunk_index"],
                        "hybrid_score": c.get("hybrid_score", 0),
                        "text_preview": c["text"][:150] + "..."
                    }
                    for c in chunks
                ]
            }

        # Get answers from both policies in isolation
        policy_1_result = get_policy_answer(doc_id_1)
        policy_2_result = get_policy_answer(doc_id_2)

        # Build comparison summary
        summary_prompt = f"""You are an insurance policy comparison assistant.

Compare these two policy responses for the query: "{query}"

Policy 1 ({doc_id_1}):
{policy_1_result['answer']}

Policy 2 ({doc_id_2}):
{policy_2_result['answer']}

RULES:
- Compare ONLY what is explicitly stated in each policy's answer above.
- If a policy says "This information is not mentioned in the document", state that clearly for that policy.
- Do NOT infer or add information that isn't in the answers above.
- Be factual and concise — 2 to 3 sentences maximum.

COMPARISON:"""

        summary = llm.generate(summary_prompt)

        return {
            "query": query,
            "policy_1": {
                "doc_id": doc_id_1,
                **policy_1_result
            },
            "policy_2": {
                "doc_id": doc_id_2,
                **policy_2_result
            },
            "comparison_summary": summary,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


def delete_doc(doc_id: str) -> dict:
    """Delete a document from both Pinecone and BM25."""
    try:
        delete_document(doc_id)
        logger.info(f"Document deleted: {doc_id}")
        return {
            "status": "success",
            "deleted": doc_id,
            "note": "Re-ingest remaining documents to rebuild BM25 index"
        }
    except Exception as e:
        logger.error(f"Delete failed: {e}")
        raise