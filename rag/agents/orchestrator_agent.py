"""Orchestrator Agent.

Responsibility: on every turn, classify intent (ingest / retrieve / clarify /
chit-chat), resolve follow-ups against conversation state, and route to the right
sub-agent. When intent is genuinely ambiguous it asks a clarifying question rather
than guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ..schema import AgentTrace
from ..text_utils import tokenize


class Intent(str, Enum):
    INGEST = "ingest"
    RETRIEVE = "retrieve"
    CLARIFY = "clarify"
    CHITCHAT = "chitchat"


@dataclass
class Turn:
    role: str      # "user" | "assistant"
    text: str


@dataclass
class Decision:
    intent: Intent
    resolved_query: str
    message: str = ""                       # used for clarify / chitchat
    ingest_path: str | None = None
    traces: list[AgentTrace] = field(default_factory=list)


_INGEST_RE = re.compile(
    r"\b(ingest|index|re-?index|upsert|upload|add)\b.*\b(doc|document|file|corpus|folder|pdf|docx|md|txt)\b"
    r"|\.(pdf|docx|md|txt)\b",
    re.IGNORECASE,
)
_PATH_RE = re.compile(r"[\w./\\-]+\.(?:pdf|docx|md|txt)", re.IGNORECASE)
_GREETING = {"hi", "hello", "hey", "thanks", "thank", "thankyou", "bye", "goodbye", "yo", "sup"}
_FOLLOWUP_RE = re.compile(r"^(what about|how about|and|also|what else|that|it|this|they|those)\b", re.IGNORECASE)
_PRONOUN = {"it", "its", "that", "this", "they", "them", "those", "these", "one"}


class OrchestratorAgent:
    def __init__(self):
        self.history: list[Turn] = []
        self.last_topic: str | None = None

    def reset(self) -> None:
        self.history.clear()
        self.last_topic = None

    def observe_user(self, text: str) -> None:
        self.history.append(Turn("user", text))

    def observe_assistant(self, text: str) -> None:
        self.history.append(Turn("assistant", text))

    # ------------------------------------------------------------------ #
    def classify(self, text: str) -> Decision:
        raw = text.strip()
        low = raw.lower()
        content = tokenize(raw, stem=False)                 # stop-word-filtered content tokens
        all_tokens = tokenize(raw, keep_stop=True, stem=False)

        # 1. chit-chat / greetings (social words only, no real content, not a question)
        if content and all(t in _GREETING for t in content) and "?" not in raw:
            return Decision(Intent.CHITCHAT, raw,
                            message="Hi! Ask me a question about the knowledge base, "
                                    "or tell me to ingest a document.",
                            traces=[AgentTrace("OrchestratorAgent", "classify", "intent=chitchat")])

        # 2. ingestion request
        if _INGEST_RE.search(raw):
            path = None
            m = _PATH_RE.search(raw)
            if m:
                path = m.group(0)
            return Decision(Intent.INGEST, raw, ingest_path=path,
                            traces=[AgentTrace("OrchestratorAgent", "classify",
                                               f"intent=ingest path={path}")])

        # 3. follow-up resolution against conversation state
        resolved = raw
        is_followup = bool(_FOLLOWUP_RE.match(low)) or (len(content) <= 2 and any(t in _PRONOUN for t in all_tokens))
        if is_followup and self.last_topic:
            resolved = f"{raw} (regarding: {self.last_topic})"
            return Decision(Intent.RETRIEVE, resolved,
                            traces=[AgentTrace("OrchestratorAgent", "resolve_followup",
                                               f"linked to prior topic: '{self.last_topic}'"),
                                    AgentTrace("OrchestratorAgent", "classify", "intent=retrieve")])

        # 4. ambiguity -> clarify (too short / no content words / dangling pronoun)
        if len(content) == 0 or (len(content) <= 1 and any(t in _PRONOUN for t in all_tokens)):
            return Decision(Intent.CLARIFY, raw,
                            message="Could you say a bit more about what you'd like to know? "
                                    "For example, name the topic or document you're asking about.",
                            traces=[AgentTrace("OrchestratorAgent", "classify",
                                               "intent=clarify (ambiguous/underspecified)")])

        # 5. default: retrieval / QA
        return Decision(Intent.RETRIEVE, resolved,
                        traces=[AgentTrace("OrchestratorAgent", "classify", "intent=retrieve")])

    def remember_topic(self, query: str) -> None:
        """Store a compact topic string from a successful retrieval turn."""
        content = tokenize(query)
        self.last_topic = " ".join(content[:8]) if content else query
