# Multi-Agent RAG Framework — Document Ingestion & Retrieval

A compact, dependency-light reference implementation of a **four-agent
Retrieval-Augmented Generation (RAG)** system. It ingests unstructured documents
into a vector database and answers questions about them through a conversational
assistant that **cites its sources** and **checks itself for hallucination**.

It runs **fully offline with no API keys** by default, and upgrades to real
Qdrant / Google Gemini backends by setting a few environment variables.

> New to the project? Read **[PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md)** for a
> beginner-friendly walkthrough of every concept and file.

---

## The four agents

| Agent | Responsibility | Code |
|-------|----------------|------|
| **Orchestrator** | Classifies intent (ingest / retrieve / clarify / chit-chat), resolves follow-ups, manages multi-turn conversation state. | `rag/agents/orchestrator_agent.py` |
| **Ingestion** | load → clean → chunk → embed → upsert into the vector DB with metadata. Idempotent. | `rag/agents/ingestion_agent.py` |
| **Retrieval** | Embed query → hybrid search (semantic + BM25) → RRF fusion → re-rank → permission filter → top-k. | `rag/agents/retrieval_agent.py` |
| **Synthesis** | Grounded answer with `[S#]` citations + hallucination check + graceful "no answer". | `rag/agents/synthesis_agent.py` |

See **[ARCHITECTURE.md](ARCHITECTURE.md)** for the one-page diagram and data flow.

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
```

Minimum to run offline: `numpy` + `python-docx` (to read the sample `.docx`).
`qdrant-client` is optional — without it the system uses a pure-Python in-memory
vector store automatically.

### 2. Ask a question (Python)

```python
from rag.pipeline import RAGSystem

rag = RAGSystem()              # build the four agents
rag.bootstrap("recursive")     # ingest the sample corpus into a collection

turn = rag.chat("Which encryption standard is used for data at rest?")
print(turn.answer)             # -> "Data at rest is encrypted with AES-256. [S1] ..."
print(turn.citations)          # -> [{marker:'S1', source:'security_policy.md', ...}]
print(turn.faithfulness)       # -> 1.0  (fully grounded)
```

### 3. Run the evaluation

```python
from rag.config import SETTINGS
from rag.evaluation import compare_configs

for r in compare_configs(["fixed", "recursive"], "eval/test_questions.json", SETTINGS):
    print(r.config, "hit_rate=", r.hit_rate, "answer_acc=", r.answer_accuracy)
```

Full results and interpretation are in **[EVALUATION_REPORT.md](EVALUATION_REPORT.md)**.

---

## Configuration

Everything is controlled by environment variables (see `rag/config.py`). Defaults
are offline; override to use cloud backends:

| Variable | Default | Purpose |
|----------|---------|---------|
| `RAG_CORPUS_DIR` | `corpus` | Folder of source documents. |
| `RAG_EMBED_BACKEND` | `hashing` | `hashing` (offline) or `gemini`. |
| `RAG_VECTOR_BACKEND` | `qdrant` | `qdrant` (in-memory) or `memory`. |
| `RAG_QDRANT_URL` | _(unset)_ | Point at a real Qdrant cluster. |
| `RAG_LLM_BACKEND` | `offline` | `offline` (extractive) or `gemini`. |
| `RAG_DEFAULT_STRATEGY` | `recursive` | `recursive` or `fixed` chunking. |
| `RAG_TOP_K` | `4` | Chunks returned after re-ranking. |
| `GOOGLE_API_KEY` | _(unset)_ | Required for any `gemini` backend. |

To use the full PDF-specified stack (Qdrant + Gemini):

```bash
export RAG_QDRANT_URL="http://localhost:6333"
export RAG_EMBED_BACKEND="gemini"
export RAG_LLM_BACKEND="gemini"
export GOOGLE_API_KEY="your-key"
```

---

## Project layout

```
multi-agent-rag/
├── rag/
│   ├── pipeline.py            # wires the 4 agents into one chat turn (start here)
│   ├── config.py              # all settings / env vars
│   ├── schema.py              # Document / Chunk / Citation + metadata schema
│   ├── agents/
│   │   ├── orchestrator_agent.py
│   │   ├── ingestion_agent.py
│   │   ├── retrieval_agent.py
│   │   └── synthesis_agent.py
│   ├── loaders.py             # md / txt / docx / pdf loaders + cleaning
│   ├── chunking.py            # fixed + recursive strategies
│   ├── embeddings.py          # offline TF-IDF + optional Gemini
│   ├── bm25.py                # keyword search
│   ├── rerank.py              # cross-encoder-style re-ranker
│   ├── vector_store.py        # Qdrant + in-memory fallback
│   ├── llm.py                 # offline + optional Gemini
│   ├── text_utils.py          # shared tokeniser/stemmer
│   └── evaluation.py          # metrics harness
├── api/
│   └── server.py              # FastAPI wrapper around RAGSystem.chat()
├── ui/                        # custom React chat app (Vite)
│   ├── src/App.jsx            #   chat UI: citation chips, status badge, roles
│   ├── src/api.js             #   fetch wrapper for the backend
│   └── vite.config.js         #   dev server + /api proxy to port 8000
├── corpus/                    # 6 sample documents ("Northwind")
├── eval/test_questions.json   # 13 gold Q&A (incl. 1 unanswerable)
├── requirements.txt
├── README.md                  # this file
├── PROJECT_EXPLAINED.md       # beginner walkthrough
├── ARCHITECTURE.md            # one-page architecture diagram
└── EVALUATION_REPORT.md       # measured results + analysis
```

---

## Key properties

- **Permission-aware retrieval** — a `viewer` role only ever sees `public`
  chunks; `admin`/`owner` can see `confidential` ones. Set via `RAGSystem(role=...)`.
- **Idempotent ingestion** — chunk IDs are content hashes, so re-running never
  duplicates.
- **Cannot hallucinate (offline mode)** — the extractive synthesiser quotes real
  sentences; a per-sentence support check reports a `faithfulness` score.
- **Two chunking strategies in separate collections** for fair comparison.

---

## Web chat UI

A custom **React** app (`ui/`) backed by a **FastAPI** server (`api/server.py`)
provides the chat interface: answers with **inline citation chips**, a
"show sources" panel, an **ingestion-status badge**, an **⬆ Upload doc** button
(index a file straight from your computer), and a **role selector** that
demonstrates permission-aware retrieval live.

**Run it (two terminals):**

```bash
# Terminal 1 — backend (from project root)
pip install -r requirements.txt
python -m uvicorn api.server:app --reload --port 8000

# Terminal 2 — frontend
cd ui
npm install      # first time only
npm run dev      # open http://localhost:5173
```

The Vite dev server proxies `/api/*` to the backend on port 8000, so there are no
CORS steps. See section 7.5 of [PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md) for a
full walkthrough.

## Status

**All 8 case-study deliverables are complete and verified** — the four agents,
the FastAPI backend, the React UI, plus the evaluation report and architecture
diagram. See the Submission Checklist in
[PROJECT_EXPLAINED.md](PROJECT_EXPLAINED.md).
