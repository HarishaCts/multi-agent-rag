"""Core data structures and the metadata schema that travels with every chunk."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class DocMetadata:
    """Metadata schema attached to every document and copied onto every chunk.

    These fields travel into the vector store so the Retrieval Agent can filter by
    source/permissions and the Synthesis Agent can build citations.
    """

    source: str          # filename, e.g. "security_policy.md"
    title: str           # human-readable document title
    author: str          # e.g. "Security Team"
    date: str            # ISO date, e.g. "2026-02-03"
    access: str          # permission tag: public | internal | confidential
    doc_type: str        # md | txt | docx | pdf

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Document:
    """A loaded, cleaned document before chunking."""

    doc_id: str
    text: str
    metadata: DocMetadata


@dataclass
class Chunk:
    """A single chunk ready to be embedded and upserted into the vector store."""

    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int
    strategy: str                       # "fixed" | "recursive"
    metadata: DocMetadata
    embedding: list[float] | None = None

    def payload(self) -> dict[str, Any]:
        """Flat payload stored alongside the vector in the DB."""
        p = self.metadata.as_dict()
        p.update(
            {
                "chunk_id": self.chunk_id,
                "doc_id": self.doc_id,
                "text": self.text,
                "chunk_index": self.chunk_index,
                "strategy": self.strategy,
            }
        )
        return p


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, with scoring provenance for transparency."""

    chunk_id: str
    doc_id: str
    text: str
    source: str
    access: str
    metadata: dict[str, Any]
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def final_score(self) -> float:
        return self.rerank_score or self.fused_score


@dataclass
class Citation:
    marker: str          # e.g. "S1"
    source: str          # filename
    chunk_id: str
    snippet: str


@dataclass
class AgentTrace:
    """A single step in the multi-agent pipeline, logged for transparency/eval."""

    agent: str
    action: str
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


def stable_id(*parts: str) -> str:
    """Deterministic id from parts — used to make ingestion idempotent."""
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return h[:16]
