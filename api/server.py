"""FastAPI wrapper around RAGSystem — the backend for the React chat UI.

One RAGSystem instance is built and the sample corpus is ingested at startup, so
the API is ready to answer immediately. Endpoints:

    GET  /api/health            -> liveness probe
    GET  /api/status            -> indexed chunk count, strategy, collection, docs
    POST /api/chat   {message, role?}  -> one conversational turn (answer+citations)
    POST /api/ingest {path?}    -> re-index corpus, or ingest a single path
    POST /api/reset             -> clear conversation state (new chat session)

Run it:
    uvicorn api.server:app --reload --port 8000

This is a single-user demo server: it holds one shared RAGSystem and conversation
history, which is fine for local evaluation of the case study.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os
from dataclasses import asdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.agents.retrieval_agent import ROLE_ACCESS
from rag.config import SETTINGS
from rag.loaders import _LOADERS
from rag.pipeline import INGEST_ROLES, RAGSystem

# Shared state, populated at startup.
STATE: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the pipeline and ingest the corpus once, before serving requests."""
    rag = RAGSystem(settings=SETTINGS, role="admin")
    report = rag.bootstrap(SETTINGS.default_strategy)
    STATE["rag"] = rag
    STATE["last_ingest"] = {
        "documents": report.documents,
        "chunks": report.chunks,
        "collection": report.collection,
        "doc_names": report.doc_names,
    }
    yield
    STATE.clear()


app = FastAPI(title="Multi-Agent RAG API", version="1.0.0", lifespan=lifespan)

# Allow the Vite dev server (and any localhost origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _rag() -> RAGSystem:
    return STATE["rag"]  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Request/response models
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    message: str
    role: str = "admin"          # viewer | analyst | admin | owner


class IngestRequest(BaseModel):
    path: str | None = None      # None => re-index the whole sample corpus
    role: str = "viewer"         # ingestion requires admin|owner


def _require_ingest_role(role: str) -> None:
    """Reject ingestion attempts from roles without permission (defense in depth)."""
    if role.lower() not in INGEST_ROLES:
        raise HTTPException(
            status_code=403,
            detail=f"Adding documents requires one of {sorted(INGEST_ROLES)}; "
                   f"'{role}' is not permitted.",
        )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/status")
def status():
    rag = _rag()
    # Derive the true document set from the collection itself, so the count stays
    # correct after incremental uploads (not just the last ingest batch).
    payloads = rag.store.all_payloads(rag.active_collection)
    doc_names = sorted({p.get("source", "?") for p in payloads})
    return {
        "indexed": rag.indexed_count(),
        "documents": len(doc_names),
        "doc_names": doc_names,
        "strategy": rag.active_strategy,
        "collection": rag.active_collection,
        "roles": list(ROLE_ACCESS.keys()),
        "ingest_roles": sorted(INGEST_ROLES),  # roles allowed to add documents
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    rag = _rag()
    # Per-request role so the UI can demo permission-aware retrieval.
    rag.role = req.role if req.role in ROLE_ACCESS else "viewer"
    turn = rag.chat(req.message)
    return asdict(turn)


@app.post("/api/ingest")
def ingest(req: IngestRequest):
    _require_ingest_role(req.role)
    rag = _rag()
    if req.path:
        report = rag.ingestion.ingest_path(req.path, rag.active_strategy)
        rag.retrieval.refresh(report.collection)
    else:
        report = rag.bootstrap(rag.active_strategy)
    STATE["last_ingest"] = {
        "documents": report.documents,
        "chunks": report.chunks,
        "collection": report.collection,
        "doc_names": report.doc_names,
    }
    return {
        "message": f"Indexed {report.documents} document(s), {report.chunks} chunk(s) "
                   f"into '{report.collection}'.",
        "indexed": rag.indexed_count(),
        **STATE["last_ingest"],
    }


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), role: str = Form("viewer")):
    """Accept a file from the browser, save it into the corpus, and ingest it.

    The uploaded file is written into the configured corpus directory (so it
    persists and is picked up by future full re-indexes), then ingested
    incrementally into the active collection. Requires an admin|owner role.
    """
    _require_ingest_role(role)
    name = os.path.basename(file.filename or "upload")
    ext = os.path.splitext(name)[1].lower()
    if ext not in _LOADERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(_LOADERS))}.",
        )

    rag = _rag()
    corpus_dir = rag.settings.corpus_dir
    os.makedirs(corpus_dir, exist_ok=True)
    dest = os.path.join(corpus_dir, name)
    with open(dest, "wb") as f:
        f.write(await file.read())

    report = rag.ingestion.ingest_path(dest, rag.active_strategy)
    rag.retrieval.refresh(report.collection)
    STATE["last_ingest"] = {
        "documents": report.documents,
        "chunks": report.chunks,
        "collection": report.collection,
        "doc_names": report.doc_names,
    }
    return {
        "message": f"Uploaded and indexed '{name}' ({report.chunks} chunk(s)) "
                   f"into '{report.collection}'.",
        "filename": name,
        "indexed": rag.indexed_count(),
        **STATE["last_ingest"],
    }


@app.post("/api/reset")
def reset():
    rag = _rag()
    rag.orchestrator.reset()
    rag.conversation_log.clear()
    return {"status": "conversation reset"}
