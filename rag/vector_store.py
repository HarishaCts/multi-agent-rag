"""Vector store abstraction with two interchangeable backends.

- QdrantVectorStore: uses qdrant-client in in-memory mode by default (no server
  needed); point it at a real cluster with RAG_QDRANT_URL.
- InMemoryVectorStore: pure-numpy fallback so the system always runs.

Both support idempotent upserts (deterministic point ids), cosine semantic
search, and metadata filtering (used for permission-aware retrieval).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import Chunk


@dataclass
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


def _point_uuid(chunk_id: str) -> str:
    """Stable UUID derived from the chunk id -> re-upsert overwrites, no dupes."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class InMemoryVectorStore:
    backend = "memory"

    def __init__(self, dim: int):
        self.dim = dim
        self._cols: dict[str, dict[str, dict[str, Any]]] = {}

    def recreate_collection(self, name: str) -> None:
        self._cols[name] = {}

    def ensure_collection(self, name: str) -> None:
        self._cols.setdefault(name, {})

    def upsert(self, name: str, chunks: list[Chunk]) -> int:
        col = self._cols.setdefault(name, {})
        for ch in chunks:
            col[ch.chunk_id] = {
                "vector": np.asarray(ch.embedding, dtype=np.float32),
                "payload": ch.payload(),
            }
        return len(col)

    def count(self, name: str) -> int:
        return len(self._cols.get(name, {}))

    def search(
        self, name: str, query_vec: list[float], limit: int,
        access_levels: list[str] | None = None,
    ) -> list[SearchHit]:
        col = self._cols.get(name, {})
        if not col:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        qn = np.linalg.norm(q) or 1.0
        hits: list[SearchHit] = []
        for cid, rec in col.items():
            if access_levels is not None and rec["payload"].get("access") not in access_levels:
                continue
            v = rec["vector"]
            denom = (np.linalg.norm(v) or 1.0) * qn
            score = float(np.dot(q, v) / denom)
            hits.append(SearchHit(cid, score, rec["payload"]))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def all_payloads(self, name: str) -> list[dict[str, Any]]:
        return [rec["payload"] for rec in self._cols.get(name, {}).values()]


class QdrantVectorStore:
    backend = "qdrant"

    def __init__(self, dim: int, url: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client import models as qm

        self.dim = dim
        self._qm = qm
        # ":memory:" gives a fully in-process Qdrant — no server to run.
        self.client = QdrantClient(url=url) if url else QdrantClient(location=":memory:")

    def recreate_collection(self, name: str) -> None:
        if self.client.collection_exists(name):
            self.client.delete_collection(name)
        self.client.create_collection(
            collection_name=name,
            vectors_config=self._qm.VectorParams(
                size=self.dim, distance=self._qm.Distance.COSINE
            ),
        )

    def ensure_collection(self, name: str) -> None:
        if not self.client.collection_exists(name):
            self.recreate_collection(name)

    def upsert(self, name: str, chunks: list[Chunk]) -> int:
        self.ensure_collection(name)
        points = [
            self._qm.PointStruct(
                id=_point_uuid(ch.chunk_id),
                vector=ch.embedding,
                payload=ch.payload(),
            )
            for ch in chunks
        ]
        if points:
            self.client.upsert(collection_name=name, points=points, wait=True)
        return self.count(name)

    def count(self, name: str) -> int:
        return int(self.client.count(collection_name=name, exact=True).count)

    def search(
        self, name: str, query_vec: list[float], limit: int,
        access_levels: list[str] | None = None,
    ) -> list[SearchHit]:
        flt = None
        if access_levels is not None:
            flt = self._qm.Filter(
                must=[
                    self._qm.FieldCondition(
                        key="access",
                        match=self._qm.MatchAny(any=list(access_levels)),
                    )
                ]
            )
        res = self.client.query_points(
            collection_name=name,
            query=query_vec,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        ).points
        return [SearchHit(h.payload["chunk_id"], float(h.score), h.payload) for h in res]

    def all_payloads(self, name: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=name, with_payload=True, limit=256, offset=offset
            )
            out.extend(p.payload for p in points)
            if offset is None:
                break
        return out


def build_vector_store(backend: str, dim: int, url: str | None = None):
    """Try the requested backend; fall back to in-memory if it can't initialise."""
    if backend == "qdrant":
        try:
            return QdrantVectorStore(dim=dim, url=url)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[vector_store] Qdrant unavailable ({exc}); using in-memory store.")
            return InMemoryVectorStore(dim=dim)
    return InMemoryVectorStore(dim=dim)
