# Multi-Agent RAG Framework — Project Explained (Beginner's Guide)

> A plain-English walkthrough of what this project is, how every piece fits
> together, how to run it, and exactly what still needs to be done to satisfy
> the case-study submission checklist.
>
> _Document generated on 2026-10-10. Based on the use-case PDF
> (`multi Agent use case.pdf`) and a full read + test-run of the code._

---

## 1. What is this project, in one sentence?

It is a **chatbot that answers questions using your own documents instead of
the internet**, built out of **four cooperating "agents"** (small specialist
programs), each doing one job in an assembly line.

This pattern has a name: **RAG = Retrieval-Augmented Generation.**

- **Retrieval** → first _find_ the most relevant pieces of your documents.
- **Augmented** → _add_ those pieces to the question as context.
- **Generation** → _write_ an answer that is grounded in (and cites) those pieces.

The "multi-agent" part means the work is split across four programs that hand
off to each other, rather than one giant function doing everything.

### Why would anyone build this?

If you dump a company's PDFs, Word docs, and wikis into a normal chatbot, it
will often **make things up** ("hallucinate"). RAG fixes that by forcing the
bot to answer **only from documents it actually retrieved**, and to **show its
sources** (citations) so you can verify every claim.

---

## 2. The big picture (how a question flows through the system)

```
                 ┌─────────────────────────────────────────────────────────┐
   You type ───▶ │  1. ORCHESTRATOR AGENT                                   │
   a message     │     "What kind of request is this?"                      │
                 │     → ingest a doc?  → ask a question?  → unclear?        │
                 └───────────────┬─────────────────────────────────────────┘
                                 │  (routes to the right agent)
          ┌──────────────────────┼───────────────────────────┐
          ▼                      ▼                            ▼
 ┌─────────────────┐   ┌───────────────────┐        "Could you clarify?"
 │ 2. INGESTION    │   │ 3. RETRIEVAL      │         (asks a question back
 │    AGENT        │   │    AGENT          │          instead of guessing)
 │ load→clean→     │   │ search the vector │
 │ chunk→embed→    │   │ DB (semantic +    │
 │ store in DB     │   │ keyword), re-rank │
 └─────────────────┘   └─────────┬─────────┘
                                 │  (top matching chunks)
                                 ▼
                       ┌───────────────────┐
                       │ 4. SYNTHESIS      │
                       │    AGENT          │
                       │ write a grounded  │
                       │ answer + [S1][S2] │
                       │ citations, check  │
                       │ for hallucination │
                       └─────────┬─────────┘
                                 ▼
                           Answer shown to you
```

Think of it like a **library help desk**:
1. **Orchestrator** = the front-desk clerk who figures out what you need.
2. **Ingestion** = the librarian who files new books onto the shelves.
3. **Retrieval** = the researcher who runs to the shelves and grabs the 4 most
   relevant pages.
4. **Synthesis** = the writer who reads those pages and writes you a short,
   sourced answer.

---

## 3. Key vocabulary (don't skip this — the rest depends on it)

| Term | Beginner meaning |
|------|------------------|
| **Corpus** | Your collection of source documents (here: 6 sample files in `corpus/`). |
| **Chunk** | A small slice of a document (a few sentences). We search chunks, not whole files, because a file is too big to match precisely. |
| **Embedding** | A list of numbers that represents the *meaning* of a chunk. Similar meanings → similar numbers. This is how the computer does "semantic" search. |
| **Vector database** | A database that stores embeddings and can instantly find the "closest in meaning" ones. Here: **Qdrant** (with a pure-Python fallback). |
| **Semantic search** | Find chunks by *meaning* (embeddings). Good for "data protection" matching "encryption". |
| **Keyword search (BM25)** | Find chunks by *exact words*. Good for codes like "AES-256" or "99.95%". |
| **Hybrid search** | Do both of the above and merge the results — best of both worlds. |
| **Re-ranking** | A second, more precise pass that re-sorts the top candidates so the single best chunk ends up at #1. |
| **Metadata** | Facts *about* a document (source, author, date, access level) stored next to every chunk. |
| **Grounding / citation** | Every sentence in the answer points back to the chunk it came from (`[S1]`, `[S2]`…). |
| **Hallucination check** | A test that verifies each answer sentence actually traces back to a retrieved chunk. |
| **Faithfulness** | A score (0–1) = what fraction of the answer is supported by real sources. 1.0 = fully grounded. |
| **Idempotent** | Running ingestion twice does **not** create duplicate chunks (same content → same ID → it overwrites). |

---

## 4. The four agents, explained one at a time

### Agent 1 — Orchestrator  (`rag/agents/orchestrator_agent.py`)
**Job:** the traffic controller. On every message it decides the **intent**:

- `ingest` → you want to add/index a document (e.g. *"index report.pdf"*).
- `retrieve` → you're asking a question (the normal case).
- `clarify` → your message is too vague, so it **asks you a question back**
  instead of guessing.
- `chitchat` → greetings like "hi"/"thanks".

It also **remembers the conversation** so follow-ups work. If you ask
*"How much is the Team edition?"* and then *"What about Enterprise?"*, it links
"what about" to the previous topic. This is the **multi-turn conversation
state** the checklist asks for.

### Agent 2 — Ingestion  (`rag/agents/ingestion_agent.py`)
**Job:** turn raw files into searchable chunks. The pipeline is
**load → clean → chunk → embed → store**:

1. **Load** — read `.md`, `.txt`, `.docx`, `.pdf` (see `rag/loaders.py`).
2. **Clean** — strip markdown symbols, fix weird characters, remove the
   metadata header lines so they don't pollute the text.
3. **Chunk** — cut into slices using **two different strategies** (`rag/chunking.py`):
   - `fixed` — equal-sized windows with overlap.
   - `recursive` — splits on paragraphs → lines → sentences (smarter boundaries).
4. **Embed** — turn each chunk into numbers (`rag/embeddings.py`).
5. **Store** — save into the vector DB with its metadata (`rag/vector_store.py`).

**Idempotent:** each chunk's ID is a hash of its content, so re-ingesting the
same corpus overwrites instead of duplicating.

### Agent 3 — Retrieval  (`rag/agents/retrieval_agent.py`)
**Job:** given a question, find the best chunks. Steps:

1. **Embed the question** (same method used during ingestion — important!).
2. **Semantic search** in the vector DB for closest-meaning chunks.
3. **Keyword search (BM25)** for exact-word matches.
4. **Fuse** both lists with **Reciprocal Rank Fusion (RRF)** — a fair way to
   merge two ranked lists.
5. **Re-rank** the survivors with a cross-encoder-style scorer for precision.
6. **Permission filter** — a `viewer` can only see `public` chunks; an `admin`
   can see `confidential` ones too. (See `ROLE_ACCESS` in the file.)
7. Return the **top-k** chunks (default 4) with all their source info.

### Agent 4 — Synthesis  (`rag/agents/synthesis_agent.py`)
**Job:** write the final answer. Two modes:

- **Offline / extractive (default)** — picks the actual sentences from the
  chunks that best answer the question. **It literally cannot hallucinate**
  because every sentence is quoted from a real chunk.
- **Gemini / abstractive (optional)** — if you provide a Google API key, an LLM
  writes a fluent answer with `[S#]` citations.

Either way it then runs the **hallucination check** (every sentence must trace
back to a chunk → `faithfulness` score) and handles the
**"I couldn't find anything about that"** case gracefully instead of making
something up.

---

## 4.5 How grounding is enforced (why it won't go "outside" your knowledge base)

A common worry with AI assistants is that they answer from their own training
data and **make things up**. This system is built so it answers **only from your
indexed documents**. That rule isn't one line of code — it's enforced in **four
layers**, so even if one layer were bypassed the others still hold.

### Layer 1 — The explicit instruction given to the LLM
When the optional Gemini LLM is enabled, the Synthesis Agent hands it a strict
system prompt — this is the literal "rule" — in
`rag/agents/synthesis_agent.py` (`_generate_abstractive`):

```python
system = (
    "You are a grounded question-answering assistant. Answer ONLY from the "
    "provided sources. Cite each claim with its [S#] marker. If the sources "
    "do not contain the answer, say you don't know."
)
```
The `context` it builds contains **only the retrieved chunks** — the model never
receives anything beyond your knowledge base.

### Layer 2 — The default mode literally cannot go outside
By default there is **no external LLM at all** (`OfflineLLM` in `rag/llm.py`
cannot generate). Instead the answer is assembled by **copying the best-matching
sentences straight out of the retrieved chunks** (`_generate_extractive` in
`rag/agents/synthesis_agent.py`). Because every sentence is quoted from a real
chunk, drifting "outside" the knowledge base is structurally impossible.

### Layer 3 — The "no relevant context → refuse" guard
Before answering, it checks whether anything retrieved is actually relevant; if
not, it **abstains** instead of inventing (`synthesize()` in
`rag/agents/synthesis_agent.py`):

```python
if not chunks or confidence < self.settings.no_answer_threshold:
    return SynthesisResult(answer=NO_CONTEXT_MESSAGE, ... no_answer=True, ...)
```

### Layer 4 — The hallucination check (verifies the rule was obeyed)
After generating, it re-checks that **every sentence traces back to a retrieved
chunk** (`_support()` in `rag/agents/synthesis_agent.py`). Any unsupported
sentence lowers the `faithfulness` score (`1.0` = fully grounded, which is what
the evaluation reports).

### And the knowledge base itself is bounded
The Synthesis Agent can only use what the **Retrieval Agent** gives it, and
retrieval only ever reads from your **vector DB** (never the web) — and further
limits results to what your role is allowed to see (`ROLE_ACCESS` in
`rag/agents/retrieval_agent.py`).

> **In one sentence:** the Retrieval Agent limits *what text exists* (your DB
> only), and the Synthesis Agent's prompt + extractive mode + no-answer guard +
> hallucination check limit *what the answer may say* (only that text). Together
> that is what "grounded, cited, no hallucination" means in this project.

---

## 5. What the supporting files do

| File | What it is for |
|------|----------------|
| `rag/pipeline.py` | **The conductor.** Wires the 4 agents together and runs one chat turn end-to-end. Start here to understand the flow. |
| `rag/config.py` | All settings in one place (chunk sizes, how many results, which backend). Controlled by environment variables. *(This is the file currently open in your editor.)* |
| `rag/schema.py` | The "shapes" of the data: `Document`, `Chunk`, `Citation`, the **metadata schema** (source/author/date/access/doc_type), and the stable-ID helper for idempotency. |
| `rag/loaders.py` | Reads each file type and cleans the text. |
| `rag/chunking.py` | The two chunking strategies. |
| `rag/embeddings.py` | Offline hashed-TF-IDF embedder (default) + optional Gemini embedder. |
| `rag/bm25.py` | The keyword-search engine. |
| `rag/rerank.py` | The precision re-ranker (+ optional real cross-encoder). |
| `rag/vector_store.py` | Qdrant backend + pure-Python in-memory fallback. |
| `rag/llm.py` | Offline (no-op) LLM + optional Gemini LLM. |
| `rag/text_utils.py` | Shared tokeniser/stemmer so every stage agrees on "what a word is". |
| `rag/evaluation.py` | The **test harness** — scores retrieval & answers, compares strategies. |
| `api/server.py` | **FastAPI backend** — a thin web wrapper around `RAGSystem.chat()` that the React UI calls (`/api/chat`, `/api/status`, `/api/ingest`, `/api/reset`). |
| `ui/` | **The custom React chat app** (Vite). Shows answers with inline citation chips, an ingestion-status badge, and a role selector. See section 7.5. |
| `scripts/inspect_db.py` | **Vector-DB inspector** — prints collections, chunk counts, per-chunk metadata, text, and embedding vectors. See section 7.6. |
| `scripts/test_gemini.py` | **Gemini connectivity/key tester** — PASS/FAIL for an embedding + generation call. See section 7.8. |
| `.env` / `.env.example` | Local secrets (gitignored) / template. Hold `GOOGLE_API_KEY` and the `RAG_*` backend switches. See section 7.8. |
| `corpus/` | 6 sample documents (a fictional product called "Northwind"). |
| `eval/test_questions.json` | 13 gold questions with expected answers & sources (incl. one deliberately unanswerable "capital of France" to test abstention). |

---

## 6. A clever design choice worth understanding

This project **runs with zero setup and no API keys by default.** It does that
by shipping "offline" versions of the three things that normally need the cloud:

| Component | Default (offline) | Optional upgrade |
|-----------|-------------------|------------------|
| Embeddings | Hashed TF-IDF (pure numpy) | Google `text-embedding-004` (`RAG_EMBED_BACKEND=gemini`) |
| Vector DB | Qdrant **in-memory** → pure-Python fallback | Real Qdrant server (`RAG_QDRANT_URL=...`) |
| Answer LLM | Extractive (quotes real sentences) | Gemini (`RAG_LLM_BACKEND=gemini` + `GOOGLE_API_KEY`) |
| Re-ranker | Lexical cross-encoder surrogate | `sentence-transformers` cross-encoder |

You flip to the "real" cloud versions by setting environment variables — no code
changes. Great for learning: it works today, and you can upgrade piece by piece.

---

## 7. How to run it yourself (step by step)

> You have **Python 3.14 and numpy** already installed, which is all you need
> for the default offline mode. `python-docx` is also installed (needed to read
> the `.docx` sample). Qdrant is **not** installed, so it automatically uses the
> pure-Python vector store — that's fine and expected.

**Step 1 — open a terminal in the project folder:**
```powershell
cd C:\Users\2516233\Downloads\multi-agent-rag
```

**Step 2 — (optional) install the full stack** so you use real Qdrant:
```powershell
pip install numpy python-docx pypdf qdrant-client
```

**Step 3 — ask a question from Python:**
```python
from rag.pipeline import RAGSystem

rag = RAGSystem()            # build the 4 agents
rag.bootstrap("recursive")   # ingest the sample corpus

turn = rag.chat("Which encryption standard is used for data at rest?")
print(turn.answer)           # grounded answer
print(turn.citations)        # where it came from
print(turn.faithfulness)     # 1.0 = fully grounded
```

**Step 4 — run the evaluation** comparing the two chunking strategies:
```python
from rag.config import SETTINGS
from rag.evaluation import compare_configs
for r in compare_configs(["fixed", "recursive"], "eval/test_questions.json", SETTINGS):
    print(r.config, "hit_rate=", r.hit_rate, "answer_acc=", r.answer_accuracy)
```

### ✅ Verified: it works right now
I ran the above on this machine. Results:
- Ingested **6 documents → 18 chunks** (recursive strategy).
- Question *"Which encryption standard is used for data at rest?"* →
  **"Data at rest is encrypted with AES-256. [S1] …"**, cited to
  `security_policy.md`, **faithfulness 1.0**, latency ~8 ms.

---

## 7.5 Running the web chat UI (React + FastAPI)

The project ships a **custom React chat app** (`ui/`) backed by a small
**FastAPI** server (`api/server.py`). This is Deliverable 6 from the case study.

### How the two pieces fit together

```
  Browser (React app, port 5173)
        │   fetch /api/chat , /api/status , /api/ingest
        ▼
  Vite dev server  ──proxies /api/*──►  FastAPI server (uvicorn, port 8000)
                                              │  calls
                                              ▼
                                        RAGSystem.chat()  → the 4 agents
```

- **`api/server.py`** builds **one** `RAGSystem` and **ingests the corpus once at
  startup**, then exposes it over HTTP:

  | Endpoint | Method | What it does |
  |----------|--------|--------------|
  | `/api/health` | GET | Liveness check. |
  | `/api/status` | GET | Indexed chunk count, strategy, collection, doc names, available roles. Drives the status badge. |
  | `/api/chat` | POST | `{message, role}` → one full turn (answer, citations, retrieved chunks, intent, faithfulness, latency). |
  | `/api/ingest` | POST | `{path?, role}` → re-index the corpus, or ingest one server-side file/folder. **403 unless role is admin/owner.** |
  | `/api/upload` | POST | **multipart file upload** (`file`, `role`) → saves the file into `corpus/` and indexes it. Backs the "Upload doc" button. **403 unless role is admin/owner.** |
  | `/api/reset` | POST | Clear conversation state (start a new chat). |

- **`ui/` (React + Vite)** — the browser app. Key files:

  | File | Purpose |
  |------|---------|
  | `ui/src/App.jsx` | The whole UI: message list, input box, role selector, re-index / new-chat buttons, status badge. |
  | `ui/src/api.js` | Tiny `fetch` wrapper for the five endpoints above. |
  | `ui/src/styles.css` | Dark-theme chat styling. |
  | `ui/vite.config.js` | Dev server on port 5173; **proxies `/api/*` to port 8000** so there are no CORS issues. |
  | `ui/package.json` | React 18 + Vite dependencies and `dev`/`build` scripts. |

### What the UI shows (Deliverable 6 requirements)

- **Inline citation chips** — `[S1]`, `[S2]` markers in the answer become little
  green chips; hover a chip to see the source filename and the exact snippet it
  came from.
- **"Show sources" panel** — expands under each answer listing every cited chunk.
- **`/help` command** — type `/help` (or click the suggestion on the welcome
  screen) and the assistant replies with **which topics it can help with** — the
  list of indexed documents your current role is allowed to see — plus example
  questions. Topics are permission-filtered, so a `viewer` sees fewer than an
  `admin`.
- **`/documents` command + "Documents" button** (admin / owner only) — lists
  **everything in the vector DB**: each document's title, source, author, date,
  access tag, doc type, chunk count, and a **per-chunk breakdown** (chunk id,
  index, size in chars, text preview). Non-admin roles get a permission message;
  the button is only shown to admin/owner. Same data as the `scripts/inspect_db.py`
  CLI (section 7.6), but from inside the chat.
- **"⬆ Upload doc" button** — pick a `.md` / `.txt` / `.docx` / `.pdf` file from
  your computer; it is saved into `corpus/`, indexed immediately, and the status
  badge updates. **Only enabled for `admin` / `owner` roles** (see below); it is
  greyed out with an explanatory tooltip for `viewer` / `analyst`.
- **Ingestion-status badge** — e.g. *"● 18 chunks indexed · 6 docs · recursive"*,
  refreshed after an upload or re-index.
- **Role selector** (viewer / analyst / admin / owner) — controls **two**
  permission behaviours live:
  1. *Retrieval* — as `viewer`, ask *"Who should I contact about a security
     breach?"* → graceful **no-answer** (the contact is in a **confidential**
     document); as `admin` → the cited answer.
  2. *Ingestion* — **only `admin` / `owner` may add documents.** As `viewer` /
     `analyst` the Upload and Re-index buttons are disabled, and typing
     `ingest …` in chat returns a polite permission-denied message.
- **Intent / faithfulness / latency tags** under each answer for transparency.

### Run it locally (two terminals)

> One-time install: `pip install -r requirements.txt` (adds FastAPI + uvicorn)
> and, in `ui/`, `npm install` (adds React + Vite). Node.js is required for the UI.

**Terminal 1 — start the backend** (from the project root):
```powershell
cd C:\Users\2516233\Downloads\multi-agent-rag
python -m uvicorn api.server:app --reload --port 8000
```
Wait for it to finish ingesting (it logs when ready). Check it:
`http://localhost:8000/api/status` should return JSON.

**Terminal 2 — start the React dev server:**
```powershell
cd C:\Users\2516233\Downloads\multi-agent-rag\ui
npm install        # first time only
npm run dev
```
Then open the URL it prints — **http://localhost:5173** — and start chatting.

**Production build** (optional): `npm run build` in `ui/` outputs static files to
`ui/dist/` which any web server (or `npm run preview`) can serve.

### ✅ Verified: the full stack works
On this machine I confirmed: `npm run build` compiles cleanly (32 modules), the
uvicorn server boots and ingests 6 docs → 18 chunks, `/api/status` and
`/api/chat` return correctly, and the `viewer` vs `admin` permission demo behaves
as described above.

---

## 7.6 How to see what's inside the vector database

**First, an important point about the default setup:** the vector DB runs in
**`:memory:` mode** — it lives *inside the Python process* and disappears when the
program stops. There is **no server and no file on disk**, so there is nothing to
open in a web browser. That's intentional: it keeps the project zero-setup and
fully offline. To inspect what's stored, use the included script — or, if the web
app is running, the **`/documents` command** (admin/owner) shows the same
information from inside the chat (section 7.5).

### The inspector script (`scripts/inspect_db.py`)

It ingests the corpus (so there's something to show) and then prints the
collection, how many chunks ("points") it holds, the metadata on each chunk, the
chunk text, and — optionally — the raw embedding numbers.

Run it from the project root:

```powershell
cd C:\Users\2516233\Downloads\multi-agent-rag

python scripts\inspect_db.py             # summary + first 3 chunks
python scripts\inspect_db.py --full      # every chunk's FULL text
python scripts\inspect_db.py --vectors   # also print the embedding vectors
```

### What the output looks like (verified on this machine)

```
======================================================================
Vector store backend : QdrantVectorStore (qdrant)
Embedding dimensions : 512
Active collection    : northwind_recursive
Documents ingested   : 6
Chunks (= points)    : 18
======================================================================

Chunks per document:
  - billing_faq.md            3 chunks   [access: public]
  - onboarding_guide.txt      3 chunks   [access: public]
  - product_overview.md       3 chunks   [access: public]
  - release_notes.docx        2 chunks   [access: public]
  - security_policy.md        3 chunks   [access: confidential]
  - sre_runbook.md            4 chunks   [access: internal]

[2] chunk_id=59e109c5c602daaf  doc_id=b6fb3ed4e35420fb  strategy=recursive
    source=product_overview.md  author=Product Team  date=2026-01-15
    access=public  type=md
    text: month. - Enterprise - unlimited users, SSO, audit logs, and a
          99.95% uptime SLA. ...
    vector[512]: [0.000, 0.000, 0.000, 0.136, 0.000, -0.087, 0.000, 0.000, …]
```

### How to read it (beginner notes)

- **"points" = chunks.** In vector-DB language each stored row is a "point": one
  embedding vector plus its payload (the metadata + text).
- **The payload** (`source`, `author`, `date`, `access`, `doc_type`, `text`,
  `strategy`) is exactly what the Retrieval Agent filters on — e.g. the `access`
  tag is what powers permission-aware retrieval.
- **The vector** (`vector[512]`) is the chunk's meaning encoded as 512 numbers;
  most are 0.0 because the offline embedder is sparse (only words present in the
  chunk get a non-zero value).
- Want a different view? `python scripts\inspect_db.py --full` prints every
  chunk's complete text — handy for checking that cleaning/chunking did the right
  thing.

> **Note:** a clickable graphical dashboard (the Qdrant web UI) is also possible,
> but it requires running Qdrant as a separate server (via Docker or Qdrant
> Cloud). Since the goal here is just to *see* what's indexed, the inspector
> script covers that without any extra installation.

---

## 7.7 How to ingest (add) documents

> **Permission:** ingesting is restricted to the **`admin`** and **`owner`**
> roles. The rule lives in one place — `INGEST_ROLES` in `rag/pipeline.py` — and
> is enforced **both** in the UI (buttons disabled; chat `ingest` request refused)
> **and** in the backend (`/api/upload` and `/api/ingest` return **HTTP 403** for
> other roles), so it can't be bypassed by calling the API directly.

Every method runs the same **load → clean → chunk → embed → store** pipeline.

**Supported file types:** `.md`, `.txt`, `.docx`, `.pdf` (anything else is skipped).

**Optional metadata header** — put these lines near the top of a document and the
loader turns them into real vector-store metadata (the `Access:` tag drives
permission filtering; it defaults to `public` if omitted):

```markdown
# My Document Title
**Source:** Finance Handbook
**Author:** Finance Team
**Date:** 2026-02-20
**Access:** confidential      ← public | internal | confidential
```

### The four ways to ingest

| # | Method | How | Role needed |
|---|--------|-----|-------------|
| 1 | **Upload button (UI)** | Click **⬆ Upload doc** in the web app, pick a file. It's saved into `corpus/` and indexed instantly. | `admin` / `owner` |
| 2 | **Re-index button (UI)** | Drop files into `corpus/` manually, then click **Re-index**. | `admin` / `owner` |
| 3 | **Chat command** | Type `ingest ./corpus/onboarding_guide.txt` in the chat box. | `admin` / `owner` |
| 4 | **Python / API** | `rag.ingestion.ingest_path("path/to/file.pdf", rag.active_strategy)` or `POST /api/upload` / `POST /api/ingest`. | API checks role (403 otherwise); direct Python calls are unrestricted. |

### `bootstrap()` vs incremental ingest — the one thing to know

- **Full re-index** (`RAGSystem.bootstrap()`, the UI **Re-index** button, or
  `POST /api/ingest` with no path) **wipes and rebuilds** the whole collection
  from the `corpus/` folder and **re-fits the embedder** on everything → the most
  consistent retrieval quality. Use after adding several documents.
- **Incremental add** (the **Upload** button, the chat command, `ingest_path()`)
  **adds to** the existing collection *without* wiping it and reuses the existing
  embedder statistics → fast, but after adding many docs a full re-index is worth
  running.

**Idempotent either way:** chunk IDs are content hashes, so re-ingesting the same
content never creates duplicates — it just overwrites.

---

## 7.8 Enabling the Gemini backend (the PDF's named LLM)

By default the system runs offline (extractive answers, hashed-TF-IDF
embeddings). You can switch it to use **Google Gemini** — the LLM named in the
case-study tech stack — for both generation and embeddings. No code changes are
needed; it's driven by a local `.env` file.

### Setup

1. Get an API key from **Google AI Studio** → https://aistudio.google.com/apikey
2. Copy `.env.example` to `.env` and fill in your key:
   ```
   GOOGLE_API_KEY=your-key-here
   RAG_LLM_BACKEND=gemini
   RAG_EMBED_BACKEND=gemini
   RAG_GEMINI_MODEL=gemini-2.0-flash
   ```
3. Install the cloud dependencies (already in requirements):
   ```powershell
   pip install -r requirements.txt
   ```
4. Verify connectivity and the key with the included tester:
   ```powershell
   python scripts\test_gemini.py
   ```
   It prints **PASS/FAIL** for an embedding call and a generation call (it never
   prints your key).

That's it — the next time you run the app or the API server, synthesis uses
Gemini (abstractive answers) and embeddings use `text-embedding-004`.

> **`.env` is gitignored** (see the repo `.gitignore`) so your key is never
> committed. Treat an API key like a password; if it's ever exposed, rotate it in
> AI Studio.

### It never breaks if Gemini is unreachable

On startup the app makes **one tiny probe call** each for embeddings and the LLM.
If they fail — no network, bad key, or a **corporate TLS-intercepting proxy**
(which raises `SSLCertVerificationError`) — it logs a clear message and
**automatically falls back to the fully-offline path**, so the app always runs.
You'll see something like:

```
[embeddings] Gemini unavailable (SSLError: ...); falling back to offline hashing embedder.
[llm] Gemini unavailable (SSLError: ...); using offline extractive synthesis.
```

**Behind a TLS proxy?** Point Python at your corporate CA bundle so the
certificate verifies, then Gemini will work:

```powershell
$env:SSL_CERT_FILE   = "C:\path\to\corporate-ca-bundle.pem"
$env:REQUESTS_CA_BUNDLE = "C:\path\to\corporate-ca-bundle.pem"
```

### Where it's wired in the code

- `rag/embeddings.py` — `GeminiEmbedder` (REST transport + timeout) and
  `build_embedder()` (probe + graceful fallback).
- `rag/llm.py` — `GeminiLLM` (REST transport + timeout) and `build_llm()`
  (probe + fallback).
- `rag/agents/synthesis_agent.py` — the abstractive path is wrapped so a runtime
  Gemini error still yields a grounded extractive answer.
- `rag/__init__.py` — loads `.env` before config reads the environment.

---

## 8. Evaluation results (measured on this machine, 13-question test set)

**Chunking-strategy comparison (full hybrid + rerank pipeline):**

| Config | Chunks | Hit-rate | Hit@1 | MRR | Source recall | Answer accuracy | Faithfulness | Abstention | p50 / p95 latency |
|--------|-------:|:-------:|:-----:|:---:|:-------------:|:---------------:|:------------:|:----------:|:-----------------:|
| **fixed** | 15 | 1.00 | 0.92 | 0.96 | 1.00 | 0.83 | 1.00 | 1.00 | 10.3 / 26.5 ms |
| **recursive** | 18 | 1.00 | 0.92 | 0.96 | 1.00 | 0.83 | 1.00 | 1.00 | 9.0 / 12.4 ms |

**Retrieval-configuration ablation (recursive chunks):**

| Config | Hit-rate | Hit@1 | MRR | Answer accuracy |
|--------|:-------:|:-----:|:---:|:---------------:|
| semantic-only | 1.00 | 1.00 | 1.00 | 0.92 |
| hybrid (no rerank) | 1.00 | 1.00 | 1.00 | 0.92 |
| hybrid + rerank | 1.00 | 0.92 | 0.96 | 0.83 |

**How to read this (beginner notes):**
- **Hit-rate = 1.00** → for every answerable question, the right source document
  was somewhere in the retrieved results. Excellent recall.
- **Faithfulness = 1.00 / Abstention = 1.00** → answers never hallucinated, and
  the "capital of France" trick question was correctly refused.
- On this **tiny 6-document corpus**, re-ranking slightly *lowers* the top-1
  metric — that's normal. Re-ranking shines on large, noisy corpora; here the
  base retrieval is already near-perfect, so there's nothing to fix and the
  re-ranker can only shuffle near-ties. **This is itself a useful finding for
  the evaluation report.**

---

## 9. Honest status check — what's done vs. what's missing

### ✅ Already implemented (strong)
- All **four agents** with clear responsibilities.
- Full **load → clean → chunk → embed → store** ingestion, **idempotent**.
- **Two chunking strategies** kept in **separate collections** for fair comparison.
- **Hybrid search (semantic + BM25) + RRF fusion + re-ranking**.
- **Permission/metadata filtering** (role-based access).
- **Grounded answers with citations + hallucination check + graceful no-answer**.
- **Multi-turn conversation state** and follow-up resolution.
- **Evaluation harness** producing recall, accuracy, faithfulness, latency.
- **Per-turn logging** of query, retrieved chunks, and answer.

### ✅ Added in this pass (previously missing)
- **`README.md`** — install + run instructions, config table, project layout.
- **`requirements.txt`** — core + optional dependency groups.
- **`EVALUATION_REPORT.md`** — written Deliverable-4 report with measured numbers,
  per-question detail, and analysis.
- **`ARCHITECTURE.md`** — one-page Deliverable-5 diagram (Mermaid) with agent
  boundaries and data flow.
- **`api/server.py` + `ui/` (React chat app)** — the Deliverable-6 chat interface:
  inline citation chips, ingestion-status badge, role selector, "show sources"
  panel. Backend verified booting + serving; frontend verified building cleanly.
  **(See section 7.5 for how to run it.)**

### ℹ️ One design note (not a gap)
- **Gemini (the PDF's chosen LLM) is wired in and switch-on via `.env`** (section
  7.8), with embeddings on `text-embedding-004` and generation on
  `gemini-2.0-flash`. The default stays offline (zero setup, cannot hallucinate),
  and if Gemini is unreachable the app **automatically falls back to offline** —
  so enabling it is safe. In your write-up, note that the full PDF stack (Gemini)
  is supported, not just the offline reference path.

---

## 10. Suggested next steps (in priority order)

All six core deliverables are now built (see the checklist below). Remaining work
is polish / optional enhancement:

1. **Export `ARCHITECTURE.md` to a PNG/SVG** if your facilitator wants a static
   image file rather than the Mermaid source.
2. _(Optional)_ Wire up **real Qdrant** and **Gemini** to match the PDF's named
   tech stack exactly (`RAG_QDRANT_URL`, `RAG_EMBED_BACKEND=gemini`,
   `RAG_LLM_BACKEND=gemini` + `GOOGLE_API_KEY`).
3. _(Optional)_ Add a few automated tests and a one-command dev script that starts
   both the API and the UI together.
4. _(Optional)_ Expand the corpus so the hybrid-search / re-rank benefit becomes
   measurable (see the evaluation report's discussion).

---

## 11. Submission Checklist (from the use-case PDF)

> From the PDF: _"Please refer to your training facilitator for detailed
> submission and evaluation instructions."_
>
> Below is the official checklist, annotated with the current status of **this**
> project. Legend: ✅ done · ⚠️ partial · ❌ not started.

| Status | Deliverable | Notes (from PDF) | Current state in this project |
|:---:|---|---|---|
| ✅ | Ingestion pipeline (load → clean → chunk → embed → store) processing the sample corpus | — | Fully implemented in `ingestion_agent.py` + `loaders.py` + `chunking.py` + `embeddings.py`; idempotent; 6 docs → 18 chunks verified. |
| ✅ | Vector DB configured (Qdrant or pgvector) with metadata-tagged chunks | — | Qdrant backend in `vector_store.py` (in-memory by default) with pure-Python fallback; every chunk carries source/author/date/access/doc_type metadata. |
| ✅ | Retrieval agent with hybrid search + re-ranking | Report recall against your test question set | `retrieval_agent.py`: semantic + BM25 + RRF + re-rank + permission filter. **Hit-rate = 1.00**, MRR = 0.96 on the 13-question set (section 8). |
| ✅ | Orchestrator agent routing ingestion vs. retrieval vs. clarification | Manages multi-turn conversation state | `orchestrator_agent.py`: intent classification + follow-up resolution + clarify-when-ambiguous + conversation history. |
| ✅ | Synthesis agent producing grounded, cited answers | Includes a hallucination check | `synthesis_agent.py`: `[S#]` citations, per-sentence support check, **faithfulness = 1.00**, graceful no-answer. |
| ✅ | **Chat assistant interface (custom React app)** | Shows citations/source snippets per answer | **`ui/` (React + Vite) + `api/server.py` (FastAPI).** Inline `[S#]` citation chips with hover snippets, "show sources" panel, ingestion-status badge, role selector. Build + server verified (section 7.5). |
| ✅ | Evaluation report comparing ≥2 chunking or retrieval configurations | Cover retrieval accuracy, faithfulness, latency | **`EVALUATION_REPORT.md`** — compares `fixed` vs `recursive` + a retrieval ablation, with per-question detail and analysis of accuracy, faithfulness, and latency. |
| ✅ | **One-page architecture diagram** | — | **`ARCHITECTURE.md`** — Mermaid diagram showing agent boundaries, data flow, and shared infrastructure (renders on GitHub; exportable to PNG/SVG). |

**Summary: 8 of 8 fully done. 🎉** The complete multi-agent RAG system — all four
agents, the FastAPI backend, and the custom React chat UI — plus every supporting
deliverable (README, requirements, evaluation report, architecture diagram) is now
in place and verified end-to-end. The only remaining items are optional polish
(static diagram export, real Qdrant/Gemini backends, automated tests).

---

### Appendix — the sample corpus

| File | Topic | Access level |
|------|-------|--------------|
| `product_overview.md` | Northwind product, editions, pricing, data sources | public |
| `billing_faq.md` | Trials, refunds, billing | public |
| `onboarding_guide.txt` | Getting started | public/internal |
| `sre_runbook.md` | On-call, backups, incident handling | internal |
| `security_policy.md` | Encryption, RBAC, incident response | **confidential** |
| `release_notes.docx` | Version changes, deprecations | public/internal |

The access levels are what make the **permission-filtering** feature
demonstrable: a `viewer` role literally cannot retrieve the confidential
security policy, while an `admin` can.
