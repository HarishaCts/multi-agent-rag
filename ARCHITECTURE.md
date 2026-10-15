# Architecture — Multi-Agent RAG Framework (one page)

A single user turn flows through four cooperating agents. The Orchestrator routes;
the other three do the work. Everything below the dashed line is shared
infrastructure (vector DB, embedder, indexes) built once at ingestion time.

```mermaid
flowchart TB
    U([User turn]) --> ORCH

    subgraph AGENTS[Multi-Agent Pipeline]
        direction TB
        ORCH["① ORCHESTRATOR AGENT<br/>classify intent · resolve follow-ups<br/>manage conversation state"]

        ORCH -->|intent = ingest| ING["② INGESTION AGENT<br/>load → clean → chunk<br/>→ embed → upsert"]
        ORCH -->|intent = retrieve| RET["③ RETRIEVAL AGENT<br/>embed query · hybrid search<br/>RRF fuse · re-rank · permission filter"]
        ORCH -->|ambiguous| CLAR([clarifying question<br/>back to user])
        ORCH -->|greeting| CHIT([chit-chat reply])

        RET -->|top-k chunks| SYN["④ SYNTHESIS AGENT<br/>grounded answer + S# citations<br/>hallucination check · no-answer guard"]
    end

    SYN --> ANS([Answer + citations<br/>shown to user])
    ING -->|status: N chunks indexed| ANS

    %% shared infrastructure
    ING -. upsert vectors + metadata .-> VDB[(Vector DB<br/>Qdrant / in-memory<br/>metadata: source·author·date·access)]
    ING -. build keyword index .-> BM25[[BM25 keyword index]]
    RET -. semantic search .-> VDB
    RET -. keyword search .-> BM25
    EMB{{Embedder<br/>offline TF-IDF / Gemini}}
    ING -. embed chunks .-> EMB
    RET -. embed query .-> EMB

    %% every turn logged for eval + UI
    ORCH -. log .-> LOG[(Conversation log<br/>query · chunks · answer)]
    SYN -. log .-> LOG

    classDef agent fill:#1a2b4a,stroke:#4a6fa5,color:#fff;
    classDef infra fill:#2d2d2d,stroke:#888,color:#fff;
    classDef io fill:#0b3d2e,stroke:#2e8b57,color:#fff;
    class ORCH,ING,RET,SYN agent;
    class VDB,BM25,EMB,LOG infra;
    class U,ANS,CLAR,CHIT io;
```

## Agent boundaries & responsibilities

| # | Agent | Input | Output | Key file |
|---|-------|-------|--------|----------|
| ① | **Orchestrator** | raw user text + history | intent + routing decision + resolved query | `rag/agents/orchestrator_agent.py` |
| ② | **Ingestion** | file path / corpus dir | chunks embedded & upserted + report | `rag/agents/ingestion_agent.py` |
| ③ | **Retrieval** | resolved query + role | top-k chunks with scores & provenance | `rag/agents/retrieval_agent.py` |
| ④ | **Synthesis** | query + retrieved chunks | grounded answer + citations + faithfulness | `rag/agents/synthesis_agent.py` |

## Data flow in words

1. **User turn** enters the **Orchestrator**, which classifies intent and (for
   follow-ups) rewrites the query using remembered conversation state.
2. **Ingest path:** the **Ingestion Agent** loads and cleans documents, splits
   them into chunks (fixed or recursive), embeds each chunk, and upserts them —
   with metadata — into the **Vector DB**, while (re)building the **BM25** index.
3. **Retrieve path:** the **Retrieval Agent** embeds the query, runs **semantic**
   (vector) and **keyword** (BM25) search, fuses the two ranked lists with
   **Reciprocal Rank Fusion**, **re-ranks** the survivors, and drops any chunk the
   user's **role** isn't permitted to see.
4. The **Synthesis Agent** turns the top-k chunks into a grounded answer with
   `[S#]` citations, verifies every sentence traces back to a chunk
   (**hallucination check**), and abstains gracefully when nothing relevant was
   found.
5. **Every turn is logged** (query → retrieved chunks → answer) for evaluation
   and for rendering in a UI.

## Shared infrastructure

- **Vector DB** — Qdrant (in-memory by default; point at a cluster via
  `RAG_QDRANT_URL`). Pure-Python fallback if `qdrant-client` is absent. Stores the
  embedding plus full metadata (`source · author · date · access · doc_type`) so
  retrieval can filter by permission.
- **Embedder** — offline hashed TF-IDF by default; swappable for Gemini
  `text-embedding-004`. The **same embedder is used at ingestion and query time**.
- **BM25 index** — in-process keyword index rebuilt after each ingestion.
- **Two collections** (`northwind_fixed`, `northwind_recursive`) keep the chunking
  strategies separate for fair comparison.

> _This Mermaid diagram renders directly on GitHub and in most Markdown viewers.
> Export to PNG/SVG (e.g. via [mermaid.live](https://mermaid.live)) if a static
> image is required for submission._
