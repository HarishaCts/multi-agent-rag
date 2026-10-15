"""Two chunking strategies so they can be compared fairly in evaluation.

- fixed:     fixed-size character windows with overlap (word-boundary aware)
- recursive: hierarchical split on paragraph -> line -> sentence -> word,
             packing pieces up to a target size with overlap (LangChain-style)
"""

from __future__ import annotations

import re

from .schema import Chunk, Document, stable_id


def _normalise_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Fixed-size windows over whitespace-normalised text, aligned to word edges."""
    words = _normalise_ws(text).split(" ")
    # approximate size in characters using a running character budget
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for w in words:
        add = len(w) + (1 if cur else 0)
        if cur_len + add > size and cur:
            chunks.append(" ".join(cur))
            # build overlap tail by characters
            if overlap > 0:
                tail: list[str] = []
                tlen = 0
                for tw in reversed(cur):
                    tadd = len(tw) + (1 if tail else 0)
                    if tlen + tadd > overlap:
                        break
                    tail.insert(0, tw)
                    tlen += tadd
                cur = tail
                cur_len = tlen
            else:
                cur, cur_len = [], 0
        cur.append(w)
        cur_len += add
    if cur:
        chunks.append(" ".join(cur))
    return [c for c in chunks if c.strip()]


_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _recursive_split(text: str, size: int, seps: list[str]) -> list[str]:
    """Recursively split on the coarsest separator that keeps pieces under `size`."""
    text = text.strip()
    if len(text) <= size or not seps:
        return [text] if text else []
    sep = seps[0]
    parts = text.split(sep) if sep else list(text)
    out: list[str] = []
    for part in parts:
        piece = part + (sep if sep and not sep.isspace() else "")
        if len(piece) <= size:
            out.append(piece.strip())
        else:
            out.extend(_recursive_split(piece, size, seps[1:]))
    return [p for p in out if p]


def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Pack recursively-split pieces up to `size`, carrying an overlap tail."""
    pieces = _recursive_split(text, size, _SEPARATORS)
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        candidate = (cur + " " + piece).strip() if cur else piece
        if len(candidate) <= size:
            cur = candidate
        else:
            if cur:
                chunks.append(cur)
            cur = (cur[-overlap:] + " " + piece).strip() if overlap and cur else piece
    if cur:
        chunks.append(cur)
    return [_normalise_ws(c) for c in chunks if c.strip()]


def chunk_document(doc: Document, strategy: str, size: int, overlap: int) -> list[Chunk]:
    """Chunk a document with the chosen strategy, producing idempotent chunk ids."""
    if strategy == "fixed":
        texts = chunk_fixed(doc.text, size, overlap)
    elif strategy == "recursive":
        texts = chunk_recursive(doc.text, size, overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    chunks: list[Chunk] = []
    for i, t in enumerate(texts):
        # id derived from content => re-running ingestion yields identical ids
        cid = stable_id(doc.doc_id, strategy, str(i), t)
        chunks.append(
            Chunk(
                chunk_id=cid,
                doc_id=doc.doc_id,
                text=t,
                chunk_index=i,
                strategy=strategy,
                metadata=doc.metadata,
            )
        )
    return chunks
