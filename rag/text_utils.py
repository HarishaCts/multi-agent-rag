"""Tokenisation, light stemming, and a tiny stopword list.

Shared by the embedder, BM25, the reranker and the synthesis agent so that all
stages agree on what a "term" is (e.g. 'encryption' and 'encrypted' collapse to
the same stem, which materially improves recall on a small corpus).
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-%][a-z0-9]+)*")

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are",
    "was", "were", "be", "been", "it", "its", "this", "that", "these", "those",
    "as", "at", "by", "with", "from", "into", "do", "does", "did", "how", "what",
    "which", "who", "when", "where", "why", "can", "will", "would", "should",
    "i", "you", "we", "they", "he", "she", "my", "your", "our", "there", "here",
}

# longest-first so 'izations' is tried before 'ion'
_SUFFIXES = ("izations", "ization", "ations", "ation", "ions", "ion",
             "ingly", "edly", "ings", "ing", "ied", "ies", "ment", "ness",
             "ed", "es", "ly", "s")


def _stem(tok: str) -> str:
    """Very light suffix stripper. Deliberately conservative; not linguistic."""
    if tok.isdigit() or any(ch in tok for ch in ".-%") or len(tok) <= 4:
        return tok
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            root = tok[: -len(suf)]
            if suf in ("ies", "ied"):
                root += "y"
            return root
    return tok


def tokenize(text: str, keep_stop: bool = False, stem: bool = True) -> list[str]:
    """Lowercase word/number tokens; keeps things like '99.95', 'aes-256', '60%'."""
    toks = _TOKEN_RE.findall(text.lower())
    if not keep_stop:
        toks = [t for t in toks if t not in STOPWORDS]
    if stem:
        toks = [_stem(t) for t in toks]
    return toks
