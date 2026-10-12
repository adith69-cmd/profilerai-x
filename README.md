![ci](https://github.com/adith69-cmd/profilerai-x/actions/workflows/ci.yml/badge.svg)

# ProfilerAI-X — Retrieval + Generation Layer

From-scratch hybrid retrieval and fine-tuned generation layer for ProfilerAI v2.
No LangChain, no LlamaIndex — BM25, dense retrieval, RRF fusion, cross-encoder
reranking, and the QLoRA fine-tune are all built from primitives.

## Status
Sessions 1–8 complete: retrieval pipeline (BM25 → dense → RRF → rerank),
fine-tune dataset synthesis, QLoRA training, base vs fine-tuned evaluation.

## Headline results

**Retrieval:** MRR 0.435 (BM25) → 0.679 (Hybrid + Rerank) on 97 MS MARCO
validation queries.

**Generation:** QLoRA fine-tune of Qwen2.5-1.5B (1.18% params trained)
improved format adherence from 0.16 to 1.00, with a documented content
regression that was partially recovered via skill-aligned data (0.093 → 0.333).

---

## Session 1 — BM25 baseline (MS MARCO v1.1, N=97)

| Metric   | Value |
|----------|-------|
| Hit@1    | 0.206 |
| Hit@5    | 0.763 |
| Hit@10   | 0.928 |
| MRR      | 0.435 |
| NDCG@10  | 0.551 |
| Fit time | 0.02 s |
| ms/query | 1.2   |

Corpus: 821 passages. **Not comparable to published MS MARCO leaderboards.**

## Session 2 — Dense + RRF hybrid

| Metric   | BM25   | Dense  | Hybrid (equal) | Hybrid (0.3/0.7) |
|----------|--------|--------|----------------|------------------|
| Hit@1    | 0.206  | 0.381  | 0.361          | 0.371            |
| Hit@5    | 0.763  | 0.918  | 0.856          | 0.887            |
| Hit@10   | 0.928  | 1.000  | 0.990          | 1.000            |
| MRR      | 0.435  | 0.577  | 0.553          | 0.573            |
| NDCG@10  | 0.551  | 0.678  | 0.652          | 0.672            |

Dense: `all-MiniLM-L6-v2`. RRF k=60. Weighted 0.3 BM25 / 0.7 Dense after
finding equal weights underperform.

## Session 3 — Cross-encoder reranking

| Metric   | Hybrid (S2) | Hybrid + Rerank | Δ      |
|----------|-------------|-----------------|--------|
| Hit@1    | 0.371       | 0.505           | +0.134 |
| Hit@5    | 0.887       | 0.959           | +0.072 |
| Hit@10   | 1.000       | 1.000           |  0.000 |
| MRR      | 0.573       | 0.679           | +0.106 |
| NDCG@10  | 0.672       | 0.753           | +0.081 |

Largest single-stage improvement in the pipeline. Cost: ~2–5 s/query on CPU.

## Session 4 — Packaging + CI
`pyproject.toml`, `pip install -e ".[dev]"`, GitHub Actions CI on every push.

## Session 5 — Interview Q/A corpus (5,002 pairs, 500 queries)

| Metric   | BM25   | Dense  | Hybrid (0.3/0.7) | Hybrid+Rerank | BM25+Rerank |
|----------|--------|--------|------------------|---------------|-------------|
| Hit@1    | 0.526  | 0.242  | 0.260            | 0.256         | **0.552**   |
| Hit@5    | 0.900  | 0.830  | 0.840            | 0.864         | **0.912**   |
| Hit@10   | 0.990  | 0.976  | 0.988            | 0.986         | 0.986       |
| MRR      | 0.653  | 0.487  | 0.507            | 0.512         | **0.674**   |
| NDCG@10  | 0.734  | 0.606  | 0.624            | 0.628         | **0.749**   |

### The pipeline that won on MS MARCO loses on this corpus
- MS MARCO: Dense > Hybrid > BM25 (short paraphrased queries).
- Interview Q/A: **BM25 > everything** (long queries, exact terms in correct doc).
- Weighted RRF inherited Dense's weakness. Reranking BM25's list directly
  gives the best MRR.

**Conclusion:** retriever choice is data-dependent. The principled fix is
fine-tuning the dense encoder on domain data — future work.

## Session 6 — Fine-tune dataset synthesis

2000 instruction examples from `Shlok307/Interview_questions` (5002 rows).
Each: `(profile + 3 reference Q/A) → 10 new questions`. Split 80/10/10.

### Design decisions
- Reference and target sets sampled disjointly (no copying).
- Profile templates: 4 per domain, generic roles.
- Target count fixed at 10 to match ProfilerAI v1 output shape.
- Assistant target = questions only (mirrors product use case).

## Session 7 — QLoRA fine-tune (v1)

- Base: `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit`
- LoRA r=16, alpha=32, target: q,k,v,o,gate,up,down proj
- 3 epochs, effective batch 8, LR 2e-4, cosine, adamw_8bit, fp16
- **Trainable: 18,464,768 / 1,562,179,072 = 1.18%**
- Train loss: 0.466, Val loss: 0.387
- Runtime: 40 min on Colab T4 (free tier)

Adapter: `huggingface.co/Adith69/profilerai-x-qwen2.5-1.5b-qlora`

## Session 8 — Base vs fine-tuned evaluation (v1)

| Metric             | Base  | Fine-tuned |
|--------------------|-------|------------|
| Format adherence   | 0.160 | **1.000**  |
| Avg # questions    | 6.1   | **10.0**   |
| Skill coverage     | **0.653** | 0.093  |

### The format/content tradeoff
The fine-tune achieved perfect format but **collapsed skill coverage**
(0.653 → 0.093). It learned to output 10 questions, not 10 relevant ones.

**Root cause:** S6 sampled references and targets independently from a
broad domain pool. Profile skills were decoupled from reference content.
The model correctly inferred the profile was decorative.

## Session 6b/8b — Skill-aligned data fix + retrain + re-eval

Regenerated S6 data with **skill-aligned sampling**: references and targets
drawn only from rows sharing a substring with profile skills.

Training v2: Train loss 0.309 (v1: 0.466), Val 0.206 (v1: 0.387).

| Metric             | Base  | FT v1 | FT v2 |
|--------------------|-------|-------|-------|
| Format adherence   | 0.160 | 1.000 | **1.000** |
| Avg # questions    | 6.1   | 10.0  | 10.0  |
| Skill coverage     | 0.653 | 0.093 | **0.333** |

### The fix worked directionally
Skill coverage **3.6× improvement** (0.093 → 0.333) with format preserved.

**Residual gap to base** (0.333 vs 0.653) attributed to:
1. Noisy substring-based skill matching in data construction.
2. 1.5B model capacity — two competing patterns (format, skill conditioning)
   cannot both be fully learned in one fine-tune.

**Decision:** not iterating a third time. The honest story
(diagnose → fix → partial recovery) is worth more than over-fitting the
data pipeline to chase a metric.

Adapter v2: `huggingface.co/Adith69/profilerai-x-qwen2.5-1.5b-qlora-v2`

## Design decisions
- BM25 k1=1.5, b=0.75 (canonical). No stemming, no stopwords.
- IDF smoothing: `log((N - df + 0.5)/(df + 0.5) + 1)` — always positive.
- RRF k=60 (Cormack 2009). Weighted 0.3/0.7.
- Cross-encoder runs on top-50 only — accuracy/latency tradeoff.
- QLoRA hyperparams: r=16, alpha=32, dropout=0 — see S7.

## Failures / lessons

### 1. BM25 ranks topically-correct passages below the MS MARCO qrel
Query "walgreens store sales average": qrel says a salary passage is
relevant; BM25's top-3 are about store sales/counts/openings. Cause:
MS MARCO qrels are single-annotator clicks. Not a code bug. Documented.

### 2. Dense beats Hybrid on MS MARCO, but loses on the interview corpus
Retriever choice is data-dependent. MS MARCO queries are paraphrased
(semantic wins); interview queries share terms verbatim (lexical wins).

### 3. First fine-tune traded content for format
Format adherence 0.16 → 1.00, skill coverage 0.653 → 0.093. Root cause
was decoupled reference/profile sampling in S6. Fixed in S6b by
skill-aligned sampling — recovered to 0.333.

## Run
    pip install -e ".[dev]"
    python scripts/eval_bm25.py        # S1
    python scripts/eval_hybrid.py      # S2
    python scripts/eval_rerank.py      # S3
    python scripts/eval_interview.py   # S5
    python scripts/build_finetune_data.py  # S6
    pytest -q

## Repo layout
    src/profilerai_x/retrieval/bm25.py      # BM25 from scratch
    src/profilerai_x/retrieval/dense.py     # Sentence-Transformers
    src/profilerai_x/retrieval/fusion.py    # RRF from scratch
    src/profilerai_x/retrieval/rerank.py    # Cross-encoder
    src/profilerai_x/retrieval/metrics.py   # Hit@K, MRR, NDCG
    src/profilerai_x/data/loaders.py        # MS MARCO loader
    src/profilerai_x/data/interview.py      # Interview Q/A loader
    scripts/build_finetune_data.py          # S6 data synthesis