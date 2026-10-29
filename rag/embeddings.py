"""Embedding backends.

Default: an offline, deterministic *hashed TF-IDF* embedder (numpy only) — no
model download, no API key, reproducible. It captures lexical + weak-semantic
similarity well enough for a training-scale corpus.

Upgrade hook: set RAG_EMBED_BACKEND=gemini (needs GOOGLE_API_KEY) to swap in
Google `text-embedding-004` without touching any other code.

Both backends expose the same interface:
    embedder.fit(texts)            # learn corpus statistics (no-op for Gemini)
    embedder.embed(text) -> list[float]
    embedder.embed_many(texts) -> list[list[float]]
    embedder.dim
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .text_utils import tokenize


def _stable_hash(token: str) -> int:
    """Process-independent hash (built-in hash() is salted by PYTHONHASHSEED)."""
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")


class HashingTfidfEmbedder:
    """Hashing-trick TF-IDF vectoriser producing L2-normalised dense vectors."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._idf = np.ones(dim, dtype=np.float64)
        self._fitted = False

    def _bucket(self, token: str) -> tuple[int, float]:
        h = _stable_hash(token)
        # separate bit for sign so collisions partially cancel (signed hashing)
        idx = (h >> 1) % self.dim
        sign = 1.0 if (h & 1) == 0 else -1.0
        return idx, sign

    def fit(self, texts: list[str]) -> "HashingTfidfEmbedder":
        """Compute per-bucket document frequency -> idf weights."""
        n_docs = max(1, len(texts))
        df = np.zeros(self.dim, dtype=np.float64)
        for t in texts:
            seen: set[int] = set()
            for tok in set(tokenize(t)):
                idx, _ = self._bucket(tok)
                seen.add(idx)
            for idx in seen:
                df[idx] += 1.0
        self._idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self._fitted = True
        return self

    def embed(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float64)
        tf: dict[int, float] = {}
        signs: dict[int, float] = {}
        for tok in tokenize(text):
            idx, sign = self._bucket(tok)
            tf[idx] = tf.get(idx, 0.0) + 1.0
            signs[idx] = sign
        for idx, raw in tf.items():
            weight = (1.0 + math.log(raw)) * self._idf[idx]   # sublinear tf * idf
            vec[idx] += signs[idx] * weight
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class GeminiEmbedder:
    """Google `text-embedding-004` backend (optional).

    Uses the REST transport (the default gRPC transport hangs behind many
    corporate proxies) and a per-request timeout so a blocked network fails fast
    instead of stalling ingestion.
    """

    def __init__(self, model: str = "models/text-embedding-004", timeout: int = 30):
        import os
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"), transport="rest")
        self._genai = genai
        self.model = model if model.startswith("models/") else f"models/{model}"
        self.timeout = timeout
        self.dim = 768

    def fit(self, texts: list[str]) -> "GeminiEmbedder":
        return self  # cloud embeddings need no corpus statistics

    def embed(self, text: str) -> list[float]:
        r = self._genai.embed_content(
            model=self.model, content=text, request_options={"timeout": self.timeout}
        )
        return r["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def build_embedder(backend: str, dim: int):
    """Build the requested embedder; if Gemini can't be reached, fall back to the
    offline hashing embedder so the system still runs (dim stays consistent)."""
    if backend == "gemini":
        try:
            emb = GeminiEmbedder()
            emb.embed("connectivity check")  # probe once so failures surface here
            return emb
        except Exception as exc:  # pragma: no cover - network/credentials dependent
            print(f"[embeddings] Gemini unavailable ({type(exc).__name__}: "
                  f"{str(exc)[:120]}); falling back to offline hashing embedder.")
            return HashingTfidfEmbedder(dim=dim)
    return HashingTfidfEmbedder(dim=dim)
