"""Evaluation harness.

Runs the gold test set against one or more configurations and reports:
- retrieval hit-rate@k and mean source-recall (did we retrieve the right doc?)
- answer accuracy (expected keyword present in the grounded answer)
- answer faithfulness (hallucination-check score from the Synthesis Agent)
- no-answer handling (out-of-corpus question answered with an abstention)
- latency per query (ms)

Used to compare >=2 chunking strategies fairly (Deliverable 4).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field, asdict
from typing import Any

from .config import Settings
from .pipeline import RAGSystem


@dataclass
class QuestionResult:
    id: str
    question: str
    answer: str
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit: bool
    hit_at_1: bool
    reciprocal_rank: float
    source_recall: float
    keyword_ok: bool
    faithfulness: float
    no_answer: bool
    correct_abstention: bool
    latency_ms: float


@dataclass
class ConfigReport:
    config: str
    collection: str
    n_docs: int
    n_chunks: int
    hit_rate: float
    hit_at_1: float
    mrr: float
    mean_source_recall: float
    answer_accuracy: float
    mean_faithfulness: float
    abstention_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    per_question: list[QuestionResult] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100) * (len(s) - 1)))))
    return round(s[k], 1)


def load_test_set(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["questions"]


def evaluate_config(strategy: str, questions: list[dict], settings: Settings,
                    role: str = "admin", use_keyword: bool = True,
                    use_rerank: bool = True, label: str | None = None) -> ConfigReport:
    rag = RAGSystem(settings=settings, role=role)
    report = rag.bootstrap(strategy)
    rag.retrieval.use_keyword = use_keyword
    rag.retrieval.use_rerank = use_rerank

    results: list[QuestionResult] = []
    latencies: list[float] = []
    for q in questions:
        rag.orchestrator.reset()                      # isolate each question
        turn = rag.chat(q["question"])
        retrieved_sources = [c["source"] for c in turn.retrieved]
        expected = q.get("expected_sources", [])
        is_noanswer_q = (q.get("expected_answer") == "NO_ANSWER") or not expected

        # rank of the first expected source among retrieved (for MRR / hit@1)
        rr, first_rank = 0.0, None
        for i, s in enumerate(retrieved_sources):
            if s in expected:
                first_rank = i + 1
                rr = 1.0 / first_rank
                break
        hit = bool(set(expected) & set(retrieved_sources)) if expected else True
        hit_at_1 = (first_rank == 1) if expected else True
        src_recall = (len(set(expected) & set(retrieved_sources)) / len(expected)) if expected else 1.0
        kw = q.get("expected_keywords", [])
        kw_ok = all(k.lower() in turn.answer.lower() for k in kw) if kw else True
        correct_abstention = (turn.no_answer == is_noanswer_q) if is_noanswer_q else True

        latencies.append(turn.latency_ms)
        results.append(QuestionResult(
            id=q["id"], question=q["question"], answer=turn.answer,
            expected_sources=expected, retrieved_sources=retrieved_sources,
            hit=hit, hit_at_1=hit_at_1, reciprocal_rank=round(rr, 3),
            source_recall=round(src_recall, 3), keyword_ok=kw_ok,
            faithfulness=round(turn.faithfulness, 3), no_answer=turn.no_answer,
            correct_abstention=correct_abstention, latency_ms=turn.latency_ms,
        ))

    answerable = [r for r in results if r.expected_sources]
    noanswer = [r for r in results if not r.expected_sources]
    mean = statistics.mean
    return ConfigReport(
        config=label or strategy,
        collection=report.collection,
        n_docs=report.documents,
        n_chunks=report.chunks,
        hit_rate=round(mean([r.hit for r in answerable]) if answerable else 0.0, 3),
        hit_at_1=round(mean([r.hit_at_1 for r in answerable]) if answerable else 0.0, 3),
        mrr=round(mean([r.reciprocal_rank for r in answerable]) if answerable else 0.0, 3),
        mean_source_recall=round(mean([r.source_recall for r in answerable]) if answerable else 0.0, 3),
        answer_accuracy=round(mean([r.keyword_ok for r in answerable]) if answerable else 0.0, 3),
        mean_faithfulness=round(mean([r.faithfulness for r in answerable]) if answerable else 0.0, 3),
        abstention_accuracy=round(mean([r.correct_abstention for r in noanswer]) if noanswer else 1.0, 3),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        per_question=results,
    )


def run_ablation(strategy: str, test_path: str, settings: Settings,
                 role: str = "admin") -> list[ConfigReport]:
    """Compare retrieval configurations on the same chunking strategy."""
    questions = load_test_set(test_path)
    variants = [
        ("semantic-only", dict(use_keyword=False, use_rerank=False)),
        ("hybrid (no rerank)", dict(use_keyword=True, use_rerank=False)),
        ("hybrid + rerank", dict(use_keyword=True, use_rerank=True)),
    ]
    return [
        evaluate_config(strategy, questions, settings, role=role, label=label, **flags)
        for label, flags in variants
    ]


def compare_configs(strategies: list[str], test_path: str, settings: Settings,
                    role: str = "admin") -> list[ConfigReport]:
    questions = load_test_set(test_path)
    return [evaluate_config(s, questions, settings, role=role) for s in strategies]


def report_to_dict(report: ConfigReport) -> dict[str, Any]:
    d = asdict(report)
    return d
