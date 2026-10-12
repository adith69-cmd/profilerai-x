"""Hybrid retrieval + cross-encoder rerank. Final Session 3 numbers."""
from __future__ import annotations
import time
from profilerai_x.data.loaders import load_msmarco
from profilerai_x.retrieval.bm25 import BM25
from profilerai_x.retrieval.dense import DenseRetriever
from profilerai_x.retrieval.fusion import reciprocal_rank_fusion
from profilerai_x.retrieval.rerank import CrossEncoderReranker
from profilerai_x.retrieval.metrics import evaluate

RETRIEVE_K = 50
FINAL_K = 10

def main(n: int = 100) -> None:
    rs = load_msmarco(n_queries=n)
    print(f"[load] corpus={len(rs.docs)} queries={len(rs.queries)}")

    print("[fit] BM25")
    bm = BM25().fit(rs.docs)
    print("[fit] Dense")
    dense = DenseRetriever().fit(rs.docs)
    print("[fit] Cross-encoder")
    reranker = CrossEncoderReranker()

    hybrid_rankings, rerank_rankings = [], []
    t0 = time.perf_counter()
    for _, q in rs.queries:
        bm_r = [d for d, _ in bm.search(q, top_k=RETRIEVE_K)]
        dn_r = [d for d, _ in dense.search(q, top_k=RETRIEVE_K)]
        fused = [d for d, _ in reciprocal_rank_fusion(
            [bm_r, dn_r], weights=[0.3, 0.7], top_k=RETRIEVE_K
        )]
        hybrid_rankings.append(fused[:FINAL_K])

        reranked = reranker.rerank(q, fused, rs.docs, top_k=FINAL_K)
        rerank_rankings.append([d for d, _ in reranked])
    print(f"[search] {time.perf_counter()-t0:.2f}s total ({n} queries)")

    gold = [rs.qrels[qid] for qid, _ in rs.queries]
    for name, r in [("Hybrid", hybrid_rankings), ("Hybrid+Rerank", rerank_rankings)]:
        m = evaluate(r, gold, ks=(1, 5, 10))
        print(f"\n[{name}]")
        for k, v in m.items():
            print(f"  {k:>10s}: {v:.4f}" if k != "n_queries" else f"  {k:>10s}: {int(v)}")

if __name__ == "__main__":
    main()