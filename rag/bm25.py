"""A small, dependency-free BM25 index for the keyword half of hybrid search."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

from .text_utils import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[dict[str, Any]] = []       # payloads, parallel to tokens
        self._tokens: list[list[str]] = []
        self._tf: list[Counter] = []
        self._df: Counter = Counter()
        self._avg_len: float = 0.0
        self._n: int = 0

    def build(self, payloads: list[dict[str, Any]]) -> "BM25Index":
        self._docs = list(payloads)
        self._tokens = [tokenize(p.get("text", "")) for p in self._docs]
        self._tf = [Counter(toks) for toks in self._tokens]
        self._df = Counter()
        for toks in self._tokens:
            for term in set(toks):
                self._df[term] += 1
        self._n = len(self._docs)
        self._avg_len = (sum(len(t) for t in self._tokens) / self._n) if self._n else 0.0
        return self

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log(1 + (self._n - df + 0.5) / (df + 0.5))

    def search(self, query: str, limit: int) -> list[tuple[dict[str, Any], float]]:
        q_terms = tokenize(query)
        scores: list[tuple[dict[str, Any], float]] = []
        for i, tf in enumerate(self._tf):
            dl = len(self._tokens[i]) or 1
            s = 0.0
            for term in q_terms:
                if term not in tf:
                    continue
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self._avg_len or 1))
                s += self._idf(term) * (freq * (self.k1 + 1)) / denom
            if s > 0:
                scores.append((self._docs[i], s))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:limit]
