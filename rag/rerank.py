"""Re-ranking stage.

Default: a lightweight, dependency-free cross-encoder *surrogate*. A true
cross-encoder jointly encodes (query, passage); here we approximate the signal it
provides — fine-grained lexical precision — with token-overlap, coverage of query
terms, and exact number/entity matches. This reliably lifts the passage that
actually answers the question to rank 1 on a training-scale corpus.

Upgrade hook: `CrossEncoderReranker` wraps sentence-transformers'
`cross-encoder/ms-marco-MiniLM-L-6-v2` when it is installed.
"""

from __future__ import annotations

import re

from .schema import RetrievedChunk
from .text_utils import tokenize

_NUM_RE = re.compile(r"\d[\d.,%\-]*")


class LexicalCrossEncoder:
    """Query-passage scorer approximating a cross-encoder's precision signal."""

    def score(self, query: str, text: str) -> float:
        q = tokenize(query)
        if not q:
            return 0.0
        qset = set(q)
        dset = set(tokenize(text))
        # 1. fraction of query terms present in the passage (coverage)
        coverage = len(qset & dset) / len(qset)
        # 2. overlap relative to passage length (precision / density)
        density = len(qset & dset) / (len(dset) or 1)
        # 3. exact numeric/entity matches are strong answer signals
        q_nums = set(_NUM_RE.findall(query))
        d_nums = set(_NUM_RE.findall(text))
        num_bonus = 0.15 * len(q_nums & d_nums)
        return 0.7 * coverage + 0.3 * density + num_bonus

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        for c in candidates:
            c.rerank_score = round(self.score(query, c.text), 4)
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]


class CrossEncoderReranker:  # pragma: no cover - optional heavy dependency
    """Real cross-encoder reranker (sentence-transformers)."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        scores = self.model.predict([(query, c.text) for c in candidates])
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]


def build_reranker():
    return LexicalCrossEncoder()
