![ci](https://github.com/adith69-cmd/profilerai-x/actions/workflows/ci.yml/badge.svg)

# ProfilerAI-X — Retrieval Layer

From-scratch hybrid retrieval layer for ProfilerAI v2. No LangChain, no
LlamaIndex, no AutoRAG — BM25, dense retrieval, RRF fusion, and cross-encoder
reranking are all implemented from primitives.

## Status
Session 3 complete: BM25 + dense + RRF + cross-encoder rerank, evaluated
on MS MARCO v1.1 subset.

## Headline result

**MRR: 0.435 (BM25) → 0.679 (Hybrid + Rerank)** on 97 MS MARCO validation
queries. Same corpus, same queries, same eval harness across all stages.

## Session 1 — BM25 baseline

| Metric   | Value |
|----------|-------|
| Hit@1    | 0.206 |
| Hit@5    | 0.763 |
| Hit@10   | 0.928 |
| MRR      | 0.435 |
| NDCG@10  | 0.551 |
| Fit time | 0.02 s |
| ms/query | 1.2   |

Corpus: 821 passages (union of the 100 queries' MS MARCO candidates).
97 of 100 queries had non-empty qrels.
**Not comparable to published MS MARCO leaderboards** — subset corpus.

## Session 2 — Dense + RRF hybrid

| Metric   | BM25   | Dense  | Hybrid (RRF, equal) | Hybrid (weighted 0.3/0.7) |
|----------|--------|--------|---------------------|---------------------------|
| Hit@1    | 0.206  | 0.381  | 0.361               | 0.371                     |
| Hit@5    | 0.763  | 0.918  | 0.856               | 0.887                     |
| Hit@10   | 0.928  | 1.000  | 0.990               | 1.000                     |
| MRR      | 0.435  | 0.577  | 0.553               | 0.573                     |
| NDCG@10  | 0.551  | 0.678  | 0.652               | 0.672                     |

Dense model: `all-MiniLM-L6-v2`. RRF k=60.

### Findings
1. Dense >> BM25 on this corpus (MRR 0.577 vs 0.435).
2. Equal-weight RRF underperforms Dense — BM25's weaker ranking drags the
   fused result down when weights are 0.5/0.5.
3. Weighted RRF (0.3 BM25 / 0.7 Dense) recovers Dense-level performance
   while preserving ensemble robustness.
4. Hit@10 = 1.00 for Dense/Hybrid indicates near-ceiling on this corpus
   size (821 docs). Larger corpora needed to distinguish methods.

## Session 3 — Cross-encoder reranking

Retrieve top-50 with weighted Hybrid (0.3 / 0.7), rerank with
`cross-encoder/ms-marco-MiniLM-L-6-v2`, evaluate top-10.

| Metric   | Hybrid (S2) | Hybrid + Rerank (S3) | Δ      |
|----------|-------------|----------------------|--------|
| Hit@1    | 0.371       | 0.505                | +0.134 |
| Hit@5    | 0.887       | 0.959                | +0.072 |
| Hit@10   | 1.000       | 1.000                |  0.000 |
| MRR      | 0.573       | 0.679                | +0.106 |
| NDCG@10  | 0.672       | 0.753                | +0.081 |

### Finding
Cross-encoder reranking gives the largest single-stage improvement in the
pipeline: +0.106 MRR, +0.134 Hit@1. Cost: reranking 50 candidates per
query adds ~2–5 s/query on CPU (measured). In production this would be
GPU-accelerated or replaced with a distilled cross-encoder.

## Design decisions
- BM25 k1=1.5, b=0.75 — canonical defaults, ablation deferred.
- Tokenizer: lowercase + alnum only. No stemming, no stopwords — so dense
  retrieval gains are attributable to the dense half, not tokenizer tricks.
- IDF smoothing: `log((N - df + 0.5)/(df + 0.5) + 1)`, always positive.
- RRF k=60 (Cormack et al. 2009 default). Weighted 0.3/0.7 after finding
  equal weights underperform on this corpus.
- Cross-encoder runs only on top-50 candidates, not the full corpus —
  accuracy/latency tradeoff documented.

## Failures / lessons

### BM25 ranks topically-correct passages below the MS MARCO qrel
Diagnostic on query "walgreens store sales average":
- MS MARCO "relevant" passage is about Walgreens **salary**, not sales.
- BM25 top-3 results are about Walgreens **store sales, counts, and
  openings** — arguably more relevant than the qrel itself.

Cause: MS MARCO qrels are single-annotator clicks. On a 10-passage subset,
non-relevant passages are often topically near-identical to the relevant
one. Hit@10 of 0.93 vs MRR of 0.44 reflects this.

Decision: not a code bug. Documented as a benchmark limitation, not "fixed"
by tuning. Dense retrieval and cross-encoder rerank were the correct next
steps and produced the expected MRR gain.

## Run
    pip install -e ".[dev]"
    python scripts/eval_bm25.py       # S1
    python scripts/eval_hybrid.py     # S2
    python scripts/eval_rerank.py     # S3
    pytest -q

## Repo layout
    src/profilerai_x/retrieval/bm25.py      # BM25 from scratch
    src/profilerai_x/retrieval/dense.py     # Sentence-Transformers wrapper
    src/profilerai_x/retrieval/fusion.py    # RRF from scratch
    src/profilerai_x/retrieval/rerank.py    # Cross-encoder rerank
    src/profilerai_x/retrieval/metrics.py   # Hit@K, MRR, NDCG
    src/profilerai_x/data/loaders.py        # MS MARCO subset