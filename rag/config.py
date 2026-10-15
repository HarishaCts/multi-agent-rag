"""Central configuration. Environment variables let you flip to cloud backends."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # --- paths ---
    corpus_dir: str = os.getenv("RAG_CORPUS_DIR", "corpus")

    # --- embeddings ---
    embedding_dim: int = int(os.getenv("RAG_EMBED_DIM", "512"))
    # "hashing" (offline, default) | "gemini" (needs GOOGLE_API_KEY)
    embedding_backend: str = os.getenv("RAG_EMBED_BACKEND", "hashing")

    # --- vector store ---
    # "qdrant" (in-memory by default) | "memory" (pure-python fallback)
    vector_backend: str = os.getenv("RAG_VECTOR_BACKEND", "qdrant")
    qdrant_url: str | None = os.getenv("RAG_QDRANT_URL")  # None => in-memory

    # --- chunking ---
    fixed_chunk_size: int = int(os.getenv("RAG_FIXED_CHUNK_SIZE", "600"))
    fixed_chunk_overlap: int = int(os.getenv("RAG_FIXED_OVERLAP", "100"))
    recursive_chunk_size: int = int(os.getenv("RAG_REC_CHUNK_SIZE", "600"))
    recursive_chunk_overlap: int = int(os.getenv("RAG_REC_OVERLAP", "100"))

    # --- retrieval ---
    candidate_k: int = int(os.getenv("RAG_CANDIDATE_K", "10"))   # per sub-search
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))                # after rerank
    rrf_k: int = 60                                              # RRF constant
    no_answer_threshold: float = float(os.getenv("RAG_NO_ANSWER_THRESHOLD", "0.12"))

    # --- llm / synthesis ---
    # "offline" (extractive, default) | "gemini"
    llm_backend: str = os.getenv("RAG_LLM_BACKEND", "offline")
    gemini_model: str = os.getenv("RAG_GEMINI_MODEL", "gemini-2.0-flash")

    # --- collections (two configs kept separate for fair comparison) ---
    default_strategy: str = os.getenv("RAG_DEFAULT_STRATEGY", "recursive")

    def collection_name(self, strategy: str) -> str:
        return f"northwind_{strategy}"


SETTINGS = Settings()
