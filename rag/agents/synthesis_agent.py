"""Synthesis (Response) Agent.

Responsibility: turn retrieved chunks + the user query into a grounded answer with
inline citations, run a hallucination check (every claim must trace back to a
retrieved chunk), and gracefully handle "no relevant context found".

Two generation modes:
- offline (default): *extractive* — picks the sentences in the retrieved chunks
  that answer the query. Cannot hallucinate; every sentence is quoted from a chunk.
- gemini: *abstractive* — prompts an LLM to write the answer with [S#] citations,
  then the same hallucination check verifies each sentence against the chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..schema import AgentTrace, Citation, RetrievedChunk
from ..text_utils import tokenize

_SENT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUM_RE = re.compile(r"\d[\d.,%\-]*")
_MARKER_RE = re.compile(r"\[S\d+\]")

NO_CONTEXT_MESSAGE = (
    "I couldn't find anything about that in the indexed knowledge base, so I don't "
    "have a grounded answer. Try rephrasing, or ingest a document that covers it."
)


@dataclass
class SynthesisResult:
    answer: str
    citations: list[Citation]
    faithfulness: float            # fraction of answer sentences supported by context
    supported: list[tuple[str, bool]]
    no_answer: bool
    traces: list[AgentTrace] = field(default_factory=list)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def _relevance(query_tokens: set[str], sentence: str, query: str) -> float:
    s_all = tokenize(sentence)
    s_tokens = set(s_all)
    if not query_tokens:
        return 0.0
    coverage = len(query_tokens & s_tokens) / len(query_tokens)
    num_bonus = 0.2 * len(set(_NUM_RE.findall(query)) & set(_NUM_RE.findall(sentence)))
    # mild length factor so bare headings don't outrank full sentences on ties
    length_factor = 0.6 + 0.4 * min(1.0, len(s_all) / 6.0)
    return coverage * length_factor + num_bonus


def _support(sentence: str, chunks: list[RetrievedChunk]) -> float:
    """Max token-containment of a sentence within any retrieved chunk (0..1)."""
    s_tokens = set(tokenize(sentence))
    if not s_tokens:
        return 1.0
    best = 0.0
    for c in chunks:
        c_tokens = set(tokenize(c.text))
        best = max(best, len(s_tokens & c_tokens) / len(s_tokens))
    return best


class SynthesisAgent:
    def __init__(self, settings, llm=None, support_threshold: float = 0.6):
        self.settings = settings
        self.llm = llm
        self.support_threshold = support_threshold

    def _best_relevance(self, query: str, chunks: list[RetrievedChunk]) -> float:
        """Scale-stable grounding signal: best query<->sentence relevance in context.

        Independent of the retrieval ranking scores, so the abstain decision stays
        consistent across retrieval configurations (semantic-only, hybrid, reranked).
        """
        q_tokens = set(tokenize(query))
        best = 0.0
        for c in chunks:
            for sent in _sentences(c.text):
                best = max(best, _relevance(q_tokens, sent, query))
        return best

    def synthesize(self, query: str, chunks: list[RetrievedChunk]) -> SynthesisResult:
        # --- no-answer guard: abstain when no retrieved sentence is relevant ---
        confidence = self._best_relevance(query, chunks)
        if not chunks or confidence < self.settings.no_answer_threshold:
            return SynthesisResult(
                answer=NO_CONTEXT_MESSAGE, citations=[], faithfulness=1.0,
                supported=[], no_answer=True,
                traces=[AgentTrace("SynthesisAgent", "no_answer",
                                   f"grounding confidence {confidence:.3f} < threshold {self.settings.no_answer_threshold}")],
            )

        # citation markers map chunk -> [S1], [S2], ...
        citations = [
            Citation(marker=f"S{i+1}", source=c.source, chunk_id=c.chunk_id,
                     snippet=(c.text[:160] + ("…" if len(c.text) > 160 else "")))
            for i, c in enumerate(chunks)
        ]

        if self.llm and getattr(self.llm, "available", False):
            try:
                answer, mode = self._generate_abstractive(query, chunks, citations), "abstractive"
            except Exception as exc:  # network/credentials failure -> stay grounded
                answer = self._generate_extractive(query, chunks, citations)
                mode = f"extractive (Gemini failed: {type(exc).__name__})"
        else:
            answer, mode = self._generate_extractive(query, chunks, citations), "extractive"

        # --- hallucination check: every answer sentence must trace to a chunk ---
        # strip citation markers so '[S1]' fragments aren't judged as claims
        supported: list[tuple[str, bool]] = []
        for sent in _sentences(_MARKER_RE.sub("", answer)):
            if len(tokenize(sent)) < 2:        # skip markers / tiny fragments
                continue
            ok = _support(sent, chunks) >= self.support_threshold
            supported.append((sent, ok))
        n = len(supported) or 1
        faithfulness = sum(1 for _, ok in supported if ok) / n

        traces = [
            AgentTrace("SynthesisAgent", "generate", f"{mode} answer over {len(chunks)} chunks"),
            AgentTrace("SynthesisAgent", "hallucination_check",
                       f"faithfulness={faithfulness:.2f} ({sum(ok for _, ok in supported)}/{n} sentences grounded)"),
        ]
        return SynthesisResult(
            answer=answer, citations=citations, faithfulness=faithfulness,
            supported=supported, no_answer=False, traces=traces,
        )

    # ------------------------------------------------------------------ #
    def _generate_extractive(self, query: str, chunks: list[RetrievedChunk],
                             citations: list[Citation]) -> str:
        """Pick the best-matching sentence(s) from the chunks and cite them."""
        q_tokens = set(tokenize(query))
        scored: list[tuple[float, str, str]] = []  # (score, sentence, marker)
        for cit, chunk in zip(citations, chunks):
            for sent in _sentences(chunk.text):
                scored.append((_relevance(q_tokens, sent, query), sent, cit.marker))
        scored.sort(key=lambda x: x[0], reverse=True)

        positives = [s for s in scored if s[0] > 0]
        if not positives:
            first = _sentences(chunks[0].text)[:1]
            return f"{first[0]} [{citations[0].marker}]" if first else chunks[0].text

        # 1st sentence = global best match
        top_score, top_sent, first_marker = positives[0]
        picked = [(top_sent, first_marker)]
        seen = {top_sent.lower()}

        # 2nd sentence = best *boosted* candidate (same-chunk continuation preferred)
        threshold = max(0.4 * top_score, 0.34)
        best = None
        for score, sent, marker in positives[1:]:
            if sent.lower() in seen:
                continue
            if len(tokenize(sent)) < 4 and not _NUM_RE.search(sent):
                continue  # skip bare headings/fragments as follow-ons
            boosted = score + (0.3 if marker == first_marker else 0.0)
            if boosted >= threshold and (best is None or boosted > best[0]):
                best = (boosted, sent, marker)
        if best:
            picked.append((best[1], best[2]))

        return " ".join(f"{sent} [{marker}]" for sent, marker in picked)

    def _generate_abstractive(self, query: str, chunks: list[RetrievedChunk],
                              citations: list[Citation]) -> str:  # pragma: no cover
        context = "\n\n".join(
            f"[{cit.marker}] (source: {cit.source})\n{chunk.text}"
            for cit, chunk in zip(citations, chunks)
        )
        system = (
            "You are a grounded question-answering assistant. Answer ONLY from the "
            "provided sources. Cite each claim with its [S#] marker. If the sources "
            "do not contain the answer, say you don't know."
        )
        user = f"Sources:\n{context}\n\nQuestion: {query}\n\nGrounded answer with [S#] citations:"
        return self.llm.generate(system, user).strip()
