"""RAGSystem — wires the four agents into one conversational pipeline.

    user turn
      -> OrchestratorAgent.classify  (intent + follow-up resolution)
         -> IngestionAgent           (if intent == ingest)
         -> RetrievalAgent           (if intent == retrieve)  -> SynthesisAgent
         -> clarify / chit-chat reply (otherwise)

Every turn is logged (query, retrieved chunks, answer, agent traces) so it can be
replayed for evaluation and rendered in the UI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .config import SETTINGS, Settings
from .embeddings import build_embedder
from .llm import build_llm
from .rerank import build_reranker
from .schema import AgentTrace
from .vector_store import build_vector_store
from .agents.ingestion_agent import IngestionAgent, IngestionReport
from .agents.orchestrator_agent import Decision, Intent, OrchestratorAgent
from .agents.retrieval_agent import ROLE_ACCESS, RetrievalAgent
from .agents.synthesis_agent import SynthesisAgent

# Only these roles may add documents to the knowledge base (ingestion).
INGEST_ROLES = {"admin", "owner"}


@dataclass
class TurnResult:
    query: str
    intent: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    faithfulness: float = 1.0
    no_answer: bool = False
    latency_ms: float = 0.0
    traces: list[dict[str, Any]] = field(default_factory=list)


class RAGSystem:
    def __init__(self, settings: Settings = SETTINGS, role: str = "admin"):
        self.settings = settings
        self.role = role
        self.embedder = build_embedder(settings.embedding_backend, settings.embedding_dim)
        self.store = build_vector_store(
            settings.vector_backend, self._embed_dim(), settings.qdrant_url
        )
        self.reranker = build_reranker()
        self.llm = build_llm(settings.llm_backend, settings.gemini_model)

        self.ingestion = IngestionAgent(self.embedder, self.store, settings)
        self.retrieval = RetrievalAgent(self.embedder, self.store, self.reranker, settings)
        self.synthesis = SynthesisAgent(settings, llm=self.llm)
        self.orchestrator = OrchestratorAgent()

        self.active_strategy = settings.default_strategy
        self.conversation_log: list[TurnResult] = []
        self._ingested = False

    def _embed_dim(self) -> int:
        return getattr(self.embedder, "dim", self.settings.embedding_dim)

    # ------------------------------------------------------------------ #
    def bootstrap(self, strategy: str | None = None) -> IngestionReport:
        """Ingest the sample corpus into the active strategy's collection."""
        strategy = strategy or self.active_strategy
        self.active_strategy = strategy
        report = self.ingestion.ingest_corpus(strategy, recreate=True)
        self.retrieval.refresh(report.collection)
        self._ingested = True
        return report

    @property
    def active_collection(self) -> str:
        return self.settings.collection_name(self.active_strategy)

    def indexed_count(self) -> int:
        return self.store.count(self.active_collection)

    def can_ingest(self) -> bool:
        """Only admin/owner roles may add documents to the knowledge base."""
        return self.role.lower() in INGEST_ROLES

    # ------------------------------------------------------------------ #
    def chat(self, user_text: str) -> TurnResult:
        """Process one conversational turn end-to-end."""
        t0 = time.perf_counter()
        self.orchestrator.observe_user(user_text)

        stripped = user_text.strip()
        if stripped.startswith("/"):
            result = self._handle_command(stripped)
        else:
            decision: Decision = self.orchestrator.classify(user_text)
            traces = list(decision.traces)

            if decision.intent == Intent.INGEST and not self.can_ingest():
                result = self._deny_ingest(user_text, traces)
            elif decision.intent == Intent.INGEST:
                result = self._handle_ingest(decision, traces)
            elif decision.intent in (Intent.CLARIFY, Intent.CHITCHAT):
                result = TurnResult(query=user_text, intent=decision.intent.value,
                                    answer=decision.message, traces=[asdict(t) for t in traces])
            else:
                result = self._handle_retrieve(decision, traces)

        result.latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.orchestrator.observe_assistant(result.answer)
        self.conversation_log.append(result)
        return result

    # ------------------------------------------------------------------ #
    def _handle_command(self, text: str) -> TurnResult:
        """Handle slash-commands (/help, /documents)."""
        cmd = text.split()[0].lower()
        if cmd in ("/help", "/topics", "/commands", "/?"):
            return self._handle_help()
        if cmd in ("/documents", "/docs", "/document"):
            return self._handle_documents()
        return TurnResult(query=text, intent="help",
                          answer=f"Unknown command '{cmd}'. Type /help to see what I can do.")

    def _handle_documents(self) -> TurnResult:
        """Admin/owner view of everything stored in the vector DB: every document,
        its metadata, and a per-chunk breakdown (id, index, size, preview)."""
        if not self.can_ingest():
            return TurnResult(
                query="/documents", intent="help",
                answer=(f"The /documents command is available to the admin and owner "
                        f"roles only. You're acting as '{self.role}'."),
            )

        payloads = self.store.all_payloads(self.active_collection)
        by_doc: dict[str, list[dict]] = {}
        for p in payloads:
            by_doc.setdefault(p.get("source", "unknown"), []).append(p)

        total_chunks = len(payloads)
        header = (
            f"Vector DB contents - collection '{self.active_collection}'\n"
            f"Backend: {type(self.store).__name__}  |  embedding dim: {self._embed_dim()}\n"
            f"{len(by_doc)} documents  |  {total_chunks} chunks (points)\n"
            + "=" * 56
        )
        if not by_doc:
            return TurnResult(query="/documents", intent="help",
                              answer=header + "\n(No documents indexed yet.)")

        blocks: list[str] = []
        for i, src in enumerate(sorted(by_doc), start=1):
            chunks = sorted(by_doc[src], key=lambda c: c.get("chunk_index", 0))
            m = chunks[0]
            lines = [
                f"\n[{i}] {m.get('title') or src}",
                f"    source: {src}  |  type: {m.get('doc_type')}  |  access: {m.get('access')}",
                f"    author: {m.get('author')}  |  date: {m.get('date')}  |  doc_id: {m.get('doc_id')}",
                f"    chunks: {len(chunks)}  ({m.get('strategy')} strategy)",
            ]
            for c in chunks:
                preview = " ".join((c.get("text") or "").split())[:70]
                lines.append(
                    f"      - idx {c.get('chunk_index')}  |  {c.get('chunk_id')}  |  "
                    f"{len(c.get('text') or '')} chars  |  \"{preview}...\""
                )
            blocks.append("\n".join(lines))

        return TurnResult(query="/documents", intent="help",
                          answer=header + "\n" + "\n".join(blocks))

    def _handle_help(self) -> TurnResult:
        """List the topics (documents) the current role is allowed to ask about."""
        allowed = ROLE_ACCESS.get(self.role.lower(), ["public"])
        payloads = self.store.all_payloads(self.active_collection)
        topics: dict[str, str] = {}
        for p in payloads:
            if p.get("access") in allowed:
                src = p.get("source", "unknown")
                topics.setdefault(src, p.get("title") or src)

        if topics:
            lines = "\n".join(
                f"  - {title}  ({src})"
                for src, title in sorted(topics.items(), key=lambda kv: kv[1].lower())
            )
            body = (f"Topics I can help with (that your '{self.role}' role can access):\n"
                    f"{lines}")
        else:
            body = "No documents are indexed yet that your role can access."

        msg = (
            "Hi! I'm a document assistant. I answer questions ONLY from the indexed "
            "knowledge base and I always cite my sources, so I won't make things up.\n\n"
            f"{body}\n\n"
            "Just ask in plain English, for example:\n"
            "  - \"Which encryption standard is used for data at rest?\"\n"
            "  - \"How much does the Team edition cost per user per month?\"\n\n"
            "Commands:  /help - show this message."
        )
        if self.can_ingest():
            msg += ("\n\nYou're acting as " + self.role + ", so you can also:\n"
                    "  - ADD documents: click the ⬆ Upload doc button, or type:  ingest <path-to-file>\n"
                    "  - INSPECT the vector DB: type  /documents  to list every document, "
                    "its metadata, and its chunks.")
        else:
            msg += "\n\n(Adding or inspecting documents requires the admin or owner role.)"
        return TurnResult(query="/help", intent="help", answer=msg)

    def _deny_ingest(self, user_text: str, traces) -> TurnResult:
        """Refuse an ingestion request from a role without ingest permission."""
        traces = traces + [AgentTrace("OrchestratorAgent", "permission_denied",
                                      f"ingest blocked for role='{self.role}'")]
        msg = (f"Adding documents requires the admin or owner role, but you're acting "
               f"as '{self.role}'. Switch your role (top-right) to admin or owner, "
               f"then try again.")
        return TurnResult(query=user_text, intent="ingest", answer=msg,
                          traces=[asdict(t) for t in traces])

    # ------------------------------------------------------------------ #
    def _handle_ingest(self, decision: Decision, traces) -> TurnResult:
        if decision.ingest_path:
            report = self.ingestion.ingest_path(decision.ingest_path, self.active_strategy)
            self.retrieval.refresh(report.collection)
            msg = (f"Ingested '{decision.ingest_path}' into collection "
                   f"'{report.collection}'. {self.indexed_count()} chunks indexed.")
        else:
            report = self.bootstrap(self.active_strategy)
            msg = (f"Re-indexed the corpus ({report.documents} documents, "
                   f"{report.chunks} chunks) into '{report.collection}'.")
        traces = traces + report.traces
        return TurnResult(query=decision.resolved_query, intent="ingest", answer=msg,
                          traces=[asdict(t) for t in traces])

    def _handle_retrieve(self, decision: Decision, traces) -> TurnResult:
        retrieval = self.retrieval.retrieve(
            decision.resolved_query, self.active_collection, role=self.role
        )
        traces = traces + retrieval.traces
        synthesis = self.synthesis.synthesize(decision.resolved_query, retrieval.chunks)
        traces = traces + synthesis.traces

        if not synthesis.no_answer:
            self.orchestrator.remember_topic(decision.resolved_query)

        retrieved = [
            {
                "chunk_id": c.chunk_id, "source": c.source, "access": c.access,
                "semantic_score": c.semantic_score, "keyword_score": c.keyword_score,
                "fused_score": round(c.fused_score, 4), "rerank_score": c.rerank_score,
                "text": c.text,
            }
            for c in retrieval.chunks
        ]
        citations = [asdict(c) for c in synthesis.citations]
        return TurnResult(
            query=decision.resolved_query, intent="retrieve", answer=synthesis.answer,
            citations=citations, retrieved=retrieved, faithfulness=synthesis.faithfulness,
            no_answer=synthesis.no_answer, traces=[asdict(t) for t in traces],
        )
