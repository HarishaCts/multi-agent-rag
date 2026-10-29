"""Multi-Agent RAG Framework for Document Ingestion & Retrieval.

A compact, dependency-light reference implementation of a four-agent RAG system:
Orchestrator, Ingestion, Retrieval, and Synthesis. Runs fully offline with
pluggable upgrade hooks (Gemini embeddings/LLM, a real cross-encoder reranker).
"""

# Load a local .env (GOOGLE_API_KEY, RAG_* backends) BEFORE config reads the
# environment. Safe no-op if python-dotenv isn't installed or no .env exists.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

__version__ = "1.0.0"
