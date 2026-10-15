"""Retrieval Agent.

Responsibility: embed the query -> run hybrid search (semantic vector search +
BM25 keyword search) -> fuse with Reciprocal Rank Fusion -> re-rank the top
candidates -> filter by permission -> return top-k chunks with provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..bm25 import BM25Index
from ..schema import AgentTrace, RetrievedChunk

# Which access tags each role may read (permission-aware retrieval).
ROLE_ACCESS = {
    "viewer": ["public"],
    "analyst": ["public", "internal"],
    "admin": ["public", "internal", "confidential"],
    "owner": ["public", "internal", "confidential"],
}


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    traces: list[AgentTrace] = field(default_factory=list)


class RetrievalAgent:
    def __init__(self, embedder, vector_store, reranker, settings):
        self.embedder = embedder
        self.store = vector_store
        self.reranker = reranker
        self.settings = settings
        self._bm25: dict[str, BM25Index] = {}
        # ablation toggles (used by the evaluation harness)
        self.use_keyword = True
        self.use_rerank = True

    def refresh(self, collection: str) -> None:
        """(Re)build the keyword index for a collection after ingestion."""
        payloads = self.store.all_payloads(collection)
        self._bm25[collection] = BM25Index().build(payloads)

    def _bm25_for(self, collection: str) -> BM25Index:
        if collection not in self._bm25:
            self.refresh(collection)
        return self._bm25[collection]

    def retrieve(self, query: str, collection: str, role: str = "admin") -> RetrievalResult:
        allowed = ROLE_ACCESS.get(role.lower(), ["public"])
        k = self.settings.candidate_k

        # --- 1. semantic (vector) search ---
        qvec = self.embedder.embed(query)
        sem_hits = self.store.search(collection, qvec, limit=k, access_levels=allowed)

        # --- 2. keyword (BM25) search, then enforce the same permission filter ---
        if self.use_keyword:
            kw_raw = self._bm25_for(collection).search(query, limit=k)
            kw_hits = [(p, s) for p, s in kw_raw if p.get("access") in allowed]
        else:
            kw_hits = []

        # --- 3. reciprocal rank fusion of the two candidate lists ---
        rrf_k = self.settings.rrf_k
        pool: dict[str, RetrievedChunk] = {}
        sem_lookup = {h.chunk_id: h.score for h in sem_hits}
        kw_lookup = {p["chunk_id"]: s for p, s in kw_hits}

        for rank, h in enumerate(sem_hits):
            rc = pool.setdefault(h.chunk_id, self._to_rc(h.payload))
            rc.semantic_score = round(h.score, 4)
            rc.fused_score += 1.0 / (rrf_k + rank + 1)
        for rank, (p, s) in enumerate(kw_hits):
            rc = pool.setdefault(p["chunk_id"], self._to_rc(p))
            rc.keyword_score = round(s, 4)
            rc.fused_score += 1.0 / (rrf_k + rank + 1)

        fused = sorted(pool.values(), key=lambda c: c.fused_score, reverse=True)
        fused = fused[: max(self.settings.top_k * 2, self.settings.top_k)]

        # --- 4. cross-encoder-style rerank -> final top-k ---
        if self.use_rerank:
            top = self.reranker.rerank(query, fused, self.settings.top_k)
            rerank_note = f"top {len(fused)}->{self.settings.top_k} after cross-encoder rerank"
        else:
            top = fused[: self.settings.top_k]
            rerank_note = "rerank disabled (ranked by fused score)"

        traces = [
            AgentTrace("RetrievalAgent", "semantic_search",
                       f"{len(sem_hits)} vector hits (role={role}, access={allowed})"),
            AgentTrace("RetrievalAgent", "keyword_search",
                       f"{len(kw_hits)} BM25 hits" + ("" if self.use_keyword else " (disabled)")),
            AgentTrace("RetrievalAgent", "fuse", f"{len(fused)} candidates after RRF"),
            AgentTrace("RetrievalAgent", "rerank", rerank_note),
        ]
        return RetrievalResult(query=query, chunks=top, traces=traces)

    @staticmethod
    def _to_rc(payload: dict) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=payload["chunk_id"],
            doc_id=payload["doc_id"],
            text=payload["text"],
            source=payload.get("source", "unknown"),
            access=payload.get("access", "public"),
            metadata=payload,
        )
