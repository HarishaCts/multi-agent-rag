"""Multi-Agent RAG Framework for Document Ingestion & Retrieval.

A compact, dependency-light reference implementation of a four-agent RAG system:
Orchestrator, Ingestion, Retrieval, and Synthesis. Runs fully offline with
pluggable upgrade hooks (Gemini embeddings/LLM, a real cross-encoder reranker).
"""

__version__ = "1.0.0"
