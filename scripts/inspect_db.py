"""Peek inside the vector database: list collections, counts, payloads, vectors.

Usage (from the project root):
    python scripts/inspect_db.py                 # summary + first chunks
    python scripts/inspect_db.py --full          # every chunk's full text
    python scripts/inspect_db.py --vectors       # also show the embedding numbers

Because the default store is in-memory, this script ingests the corpus first so
there is something to show. Point RAG_QDRANT_URL at a real Qdrant server to
inspect a persistent one instead.
"""

from __future__ import annotations

import os
import sys

# allow running as `python scripts/inspect_db.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.pipeline import RAGSystem


def main() -> None:
    full = "--full" in sys.argv
    show_vectors = "--vectors" in sys.argv

    rag = RAGSystem()
    report = rag.bootstrap(rag.active_strategy)

    store = rag.store
    print("=" * 70)
    print(f"Vector store backend : {type(store).__name__} ({getattr(store, 'backend', '?')})")
    print(f"Embedding dimensions : {rag._embed_dim()}")
    print(f"Active collection    : {rag.active_collection}")
    print(f"Documents ingested   : {report.documents}")
    print(f"Chunks (= points)    : {rag.indexed_count()}")
    print("=" * 70)

    payloads = store.all_payloads(rag.active_collection)
    # group by source document
    by_source: dict[str, int] = {}
    for p in payloads:
        by_source[p.get("source", "?")] = by_source.get(p.get("source", "?"), 0) + 1
    print("\nChunks per document:")
    for src, n in sorted(by_source.items()):
        access = next((p.get("access") for p in payloads if p.get("source") == src), "?")
        print(f"  - {src:<24} {n:>2} chunks   [access: {access}]")

    limit = len(payloads) if full else 3
    print(f"\nShowing {limit} of {len(payloads)} chunks (payload = metadata stored with each vector):\n")
    for i, p in enumerate(payloads[:limit]):
        print(f"[{i+1}] chunk_id={p['chunk_id']}  doc_id={p['doc_id']}  strategy={p.get('strategy')}")
        print(f"    source={p.get('source')}  author={p.get('author')}  "
              f"date={p.get('date')}  access={p.get('access')}  type={p.get('doc_type')}")
        text = p.get("text", "")
        preview = text if full else (text[:160] + ("…" if len(text) > 160 else ""))
        print(f"    text: {preview}")
        if show_vectors:
            vec = _vector_for(store, rag.active_collection, p["chunk_id"])
            if vec is not None:
                head = ", ".join(f"{x:.3f}" for x in vec[:8])
                print(f"    vector[{len(vec)}]: [{head}, …]")
        print()


def _vector_for(store, collection: str, chunk_id: str):
    """Best-effort fetch of one stored embedding, for either backend."""
    # pure-python in-memory store
    cols = getattr(store, "_cols", None)
    if cols is not None:
        rec = cols.get(collection, {}).get(chunk_id)
        return rec["vector"].tolist() if rec else None
    # qdrant client
    client = getattr(store, "client", None)
    if client is not None:
        import uuid
        pid = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
        pts = client.retrieve(collection_name=collection, ids=[pid], with_vectors=True)
        return list(pts[0].vector) if pts else None
    return None


if __name__ == "__main__":
    main()
