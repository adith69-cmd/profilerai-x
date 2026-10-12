"""Interview Q/A retrieval eval.

Runs four pipelines on the same corpus:
  1. BM25
  2. Dense
  3. Hybrid RRF (configurable weights)
  4. Hybrid + cross-encoder rerank
  5. BM25-only + rerank (Experiment B)

Experiment A: flip weights to [0.7, 0.3] (BM25-heavy) and rerun.
"""
from __future__ import annotations
import time
from profilerai_x.data.interview import load_interview
from profilerai_x.retrieval.bm25 import BM25
from profilerai_x.retrieval.dense import DenseRetriever
from profilerai_x.retrieval.fusion import reciprocal_rank_fusion
from profilerai_x.retrieval.rerank import CrossEncoderReranker
from profilerai_x.retrieval.metrics import evaluate

RETRIEVE_K = 50
FINAL_K = 10

# Experiment A: flip to [0.7, 0.3] to favor BM25
RRF_WEIGHTS = [0.3, 0.7]


def main(n_queries: int = 500) -> None:
    rs = load_interview(n_queries=n_queries)
    print(f"[load] corpus={len(rs.docs)} queries={len(rs.queries)}")
    print(f"[config] RRF weights = {RRF_WEIGHTS}")

    print("[fit] BM25")
    bm = BM25().fit(rs.docs)
    print("[fit] Dense")
    dense = DenseRetriever().fit(rs.docs)
    print("[fit] Cross-encoder")
    reranker = CrossEncoderReranker()

    bm_rankings, dense_rankings = [], []
    hybrid_rankings, rerank_rankings, bm_rerank_rankings = [], [], []

    t0 = time.perf_counter()
    for _, q in rs.queries:
        bm_r = [d for d, _ in bm.search(q, top_k=RETRIEVE_K)]
        dn_r = [d for d, _ in dense.search(q, top_k=RETRIEVE_K)]

        fused = [d for d, _ in reciprocal_rank_fusion(
            [bm_r, dn_r], weights=RRF_WEIGHTS, top_k=RETRIEVE_K
        )]

        reranked_hybrid = reranker.rerank(q, fused, rs.docs, top_k=FINAL_K)
        reranked_bm = reranker.rerank(q, bm_r, rs.docs, top_k=FINAL_K)

        bm_rankings.append(bm_r[:FINAL_K])
        dense_rankings.append(dn_r[:FINAL_K])
        hybrid_rankings.append(fused[:FINAL_K])
        rerank_rankings.append([d for d, _ in reranked_hybrid])
        bm_rerank_rankings.append([d for d, _ in reranked_bm])

    print(f"[search] {time.perf_counter()-t0:.2f}s ({len(rs.queries)} queries)")

    gold = [rs.qrels[qid] for qid, _ in rs.queries]
    results = [
        ("BM25",              bm_rankings),
        ("Dense",             dense_rankings),
        ("Hybrid",            hybrid_rankings),
        ("Hybrid+Rerank",     rerank_rankings),
        ("BM25+Rerank",       bm_rerank_rankings),
    ]
    for name, r in results:
        m = evaluate(r, gold, ks=(1, 5, 10))
        print(f"\n[{name}]")
        for k, v in m.items():
            if k == "n_queries":
                print(f"  {k:>10s}: {int(v)}")
            else:
                print(f"  {k:>10s}: {v:.4f}")


if __name__ == "__main__":
    main()