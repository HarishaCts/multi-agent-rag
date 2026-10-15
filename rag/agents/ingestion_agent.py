"""Ingestion Agent.

Responsibility: load raw documents -> clean -> chunk -> embed -> upsert into the
vector DB with metadata. Ingestion is idempotent: deterministic chunk ids mean
re-running on the same corpus overwrites rather than duplicates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..chunking import chunk_document
from ..loaders import load_corpus, load_document
from ..schema import AgentTrace, Chunk, Document


@dataclass
class IngestionReport:
    strategy: str
    collection: str
    documents: int
    chunks: int
    doc_names: list[str]
    traces: list[AgentTrace]


class IngestionAgent:
    def __init__(self, embedder, vector_store, settings):
        self.embedder = embedder
        self.store = vector_store
        self.settings = settings

    def _embed_chunks(self, chunks: list[Chunk]) -> None:
        vectors = self.embedder.embed_many([c.text for c in chunks])
        for c, v in zip(chunks, vectors):
            c.embedding = v

    def ingest_documents(
        self, docs: list[Document], strategy: str, fit_embedder: bool = True,
        recreate: bool = True,
    ) -> IngestionReport:
        """Chunk + embed + upsert a set of already-loaded documents."""
        if strategy == "fixed":
            size, overlap = self.settings.fixed_chunk_size, self.settings.fixed_chunk_overlap
        else:
            size, overlap = self.settings.recursive_chunk_size, self.settings.recursive_chunk_overlap

        all_chunks: list[Chunk] = []
        for doc in docs:
            all_chunks.extend(chunk_document(doc, strategy, size, overlap))

        # Fit embedder statistics on the chunk corpus (offline TF-IDF needs this).
        if fit_embedder and hasattr(self.embedder, "fit"):
            self.embedder.fit([c.text for c in all_chunks])
        self._embed_chunks(all_chunks)

        collection = self.settings.collection_name(strategy)
        if recreate:
            self.store.recreate_collection(collection)
        count = self.store.upsert(collection, all_chunks)

        traces = [
            AgentTrace("IngestionAgent", "load", f"{len(docs)} documents loaded"),
            AgentTrace("IngestionAgent", "chunk", f"{len(all_chunks)} chunks ({strategy}, size={size}, overlap={overlap})"),
            AgentTrace("IngestionAgent", "embed", f"{len(all_chunks)} embeddings (dim={self.embedder.dim})"),
            AgentTrace("IngestionAgent", "upsert", f"collection '{collection}' now holds {count} chunks"),
        ]
        return IngestionReport(
            strategy=strategy,
            collection=collection,
            documents=len(docs),
            chunks=len(all_chunks),
            doc_names=[d.metadata.source for d in docs],
            traces=traces,
        )

    def ingest_corpus(self, strategy: str, recreate: bool = True) -> IngestionReport:
        docs = load_corpus(self.settings.corpus_dir)
        return self.ingest_documents(docs, strategy, fit_embedder=True, recreate=recreate)

    def ingest_path(self, path: str, strategy: str) -> IngestionReport:
        """Add a single new document without wiping the collection (incremental)."""
        if os.path.isdir(path):
            docs = load_corpus(path)
        else:
            docs = [load_document(path)]
        # do not refit the embedder on a single doc; reuse existing corpus idf
        return self.ingest_documents(docs, strategy, fit_embedder=False, recreate=False)
