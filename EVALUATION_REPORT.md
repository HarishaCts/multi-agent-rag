# Evaluation Report — Multi-Agent RAG Framework

**Deliverable 4:** a short evaluation comparing ≥2 chunking strategies and ≥2
retrieval configurations, covering **retrieval accuracy, answer faithfulness, and
latency**.

- **Date:** 2026-10-10
- **Corpus:** 6 documents (`corpus/`, fictional "Northwind" product).
- **Test set:** 13 gold questions (`eval/test_questions.json`), including **one
  deliberately unanswerable** question ("What is the capital of France?") to test
  graceful abstention.
- **Backends used for this run:** offline hashed-TF-IDF embeddings, pure-Python
  in-memory vector store (Qdrant not installed), lexical cross-encoder re-ranker,
  extractive synthesis. Role = `admin` (full access).
- **Reproduce:** `python -c "from rag.config import SETTINGS; from rag.evaluation
  import compare_configs; [print(r) for r in compare_configs(['fixed','recursive'],
  'eval/test_questions.json', SETTINGS)]"`

---

## 1. Metrics glossary

| Metric | What it measures |
|--------|------------------|
| **Hit-rate** | Fraction of answerable questions where the correct source document appeared *anywhere* in the retrieved top-k. (Recall.) |
| **Hit@1** | Fraction where the correct source was ranked **#1**. (Precision at the top.) |
| **MRR** | Mean Reciprocal Rank — rewards putting the right source higher (1/rank). |
| **Source recall** | Of all expected source docs per question, the fraction retrieved. |
| **Answer accuracy** | Fraction of answers containing the expected keyword(s). |
| **Faithfulness** | Mean fraction of answer sentences that trace back to a retrieved chunk (hallucination check). 1.0 = no hallucination. |
| **Abstention accuracy** | For the unanswerable question, did the system correctly refuse to answer? |
| **p50 / p95 latency** | Median / 95th-percentile per-query wall-clock time. |

---

## 2. Experiment A — Chunking strategy comparison

Same full pipeline (hybrid search + re-rank), two chunking strategies, kept in
**separate collections** (`northwind_fixed`, `northwind_recursive`) for a fair test.

| Strategy | Chunks | Hit-rate | Hit@1 | MRR | Source recall | Answer acc. | Faithfulness | Abstention | p50 | p95 |
|----------|-------:|:-------:|:-----:|:---:|:-------------:|:-----------:|:------------:|:----------:|----:|----:|
| **fixed** | 15 | 1.00 | 0.92 | 0.96 | 1.00 | 0.83 | 1.00 | 1.00 | 10.3 ms | 26.5 ms |
| **recursive** | 18 | 1.00 | 0.92 | 0.96 | 1.00 | 0.83 | 1.00 | 1.00 | 9.0 ms | 12.4 ms |

**Reading it:**
- **Retrieval quality is identical** on this corpus: both strategies find the
  right document every time (hit-rate 1.00) and rank it #1 on 11 of 12 answerable
  questions (hit@1 0.92, MRR 0.96).
- **Faithfulness is perfect (1.00)** for both — the extractive synthesiser never
  introduces an unsupported sentence — and both correctly **abstain** on the
  France question.
- **The differentiator is latency stability.** `recursive` produces slightly more
  but cleaner chunks (18 vs 15) and has a **much tighter tail (p95 12.4 ms vs
  26.5 ms)** because its chunks break on natural paragraph/sentence boundaries,
  so the keyword and re-rank stages do more uniform work.

➡️ **Recommendation: use `recursive` chunking** (the current default). Equal
accuracy, better and more predictable latency.

---

## 3. Experiment B — Retrieval configuration ablation

Same `recursive` chunks; progressively enable keyword search and re-ranking.
(The `fixed` ablation gave identical figures — see §4.)

| Configuration | Hit-rate | Hit@1 | MRR | Answer acc. | Faithfulness |
|---------------|:-------:|:-----:|:---:|:-----------:|:------------:|
| semantic-only | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 |
| hybrid (semantic + BM25, no rerank) | 1.00 | 1.00 | 1.00 | 0.92 | 1.00 |
| hybrid + re-rank (full pipeline) | 1.00 | 0.92 | 0.96 | 0.83 | 1.00 |

**Reading it — an important, honest finding:**
- On this **tiny, clean 6-document corpus, semantic search alone is already
  near-perfect** (hit@1 = 1.00). There is simply no retrieval error left for
  hybrid search or re-ranking to correct.
- The lexical re-ranker very slightly **re-orders two near-tie cases** and nudges
  hit@1 from 1.00 → 0.92 (one question drops from rank 1 to rank 2). This is the
  expected behaviour of re-ranking on a corpus where base retrieval is already
  saturated: with no genuine errors to fix, it can only shuffle statistical ties.
- **This does not mean hybrid/re-rank are bad.** Their value appears on **larger,
  noisier corpora** where semantic search alone misses exact tokens (codes,
  version numbers, IDs) — exactly what BM25 + re-rank recover. On a 6-doc toy
  corpus that regime is never reached.

➡️ **Recommendation:** keep the full hybrid + re-rank pipeline (it is the robust
choice as the corpus grows), but **document that on this corpus it is not yet
necessary.** A production corpus of hundreds+ documents is where it earns its keep.

---

## 4. Per-question detail (recursive, full pipeline)

| Q | Question (abbreviated) | Hit@1 | RR | Keyword match | Faithful |
|---|------------------------|:-----:|:--:|:-------------:|:--------:|
| q1 | Team edition price per user/month | ✅ | 1.00 | ✅ | 1.00 |
| q2 | Enterprise uptime SLA | ✅ | 1.00 | ✅ | 1.00 |
| q3 | Encryption standard at rest | ✅ | 1.00 | ✅ | 1.00 |
| q4 | Free trial length & edition | ✅ | 1.00 | ✅ | 1.00 |
| q5 | Primary DB backup frequency | ✅ | 1.00 | ✅ | 1.00 |
| q6 | On-call page acknowledgement time | ✅ | 1.00 | ✅ | 1.00 |
| q7 | Natively supported data sources | ✅ | 1.00 | ⚠️ | 1.00 |
| q8 | Snowflake refresh cost reduction | ✅ | 1.00 | ✅ | 1.00 |
| q9 | Legacy v1 REST API removal date | ✅ | 1.00 | ✅ | 1.00 |
| q10 | Capital of France (unanswerable) | — | — | ✅ (abstained) | 1.00 |
| q11 | Refund on annual contract cancel | ❌ | 0.50 | ⚠️ | 1.00 |
| q12 | Security breach contact | ✅ | 1.00 | ✅ | 1.00 |
| q13 | Dashboards loading slowly | ✅ | 1.00 | ✅ | 1.00 |

**The two imperfect cases (why answer accuracy = 0.83):**
- **q7** retrieves the right document at rank 1, but the extractive answer quotes
  the sentence listing *some* data sources without the specific tokens
  "Snowflake"/"BigQuery" the keyword check looked for. Retrieval was correct; the
  single-sentence extraction was just narrow.
- **q11** ranks the correct billing doc at **#2 instead of #1** (RR 0.50), and the
  extracted sentence omits the literal "30 days" phrase. The right document *was*
  retrieved (hit-rate still counts it); only top-1 precision and the keyword
  string missed.

Both are **extraction/keyword-matching artefacts, not hallucinations** —
faithfulness stays 1.00 throughout. They point to the natural next improvement:
enabling the **abstractive (Gemini) synthesiser**, which composes a fuller answer
across chunks rather than quoting one sentence.

---

## 5. Conclusions

1. **Retrieval is excellent** on the sample corpus: 100% hit-rate, 0.96 MRR.
2. **No hallucination:** faithfulness = 1.00 everywhere, and the system correctly
   abstains on out-of-corpus questions.
3. **Chunking:** `recursive` ≥ `fixed` — equal accuracy, tighter latency. Keep it.
4. **Hybrid + re-rank** is the right production default, but this corpus is too
   small/clean to show its benefit; its value scales with corpus size and noise.
5. **Latency is excellent** (p50 ≈ 9 ms) in offline mode — expect higher numbers
   once Gemini embeddings/LLM are enabled (network-bound).
6. **Biggest quality lever next:** the abstractive Gemini synthesiser, to lift
   answer accuracy on multi-fact questions (q7, q11) without sacrificing grounding.
