"""BM25 vs Dense vs Hybrid (RRF) on the same MS MARCO subset."""
from __future__ import annotations
import time
from profilerai_x.data.loaders import load_msmarco
from profilerai_x.retrieval.bm25 import BM25
from profilerai_x.retrieval.dense import DenseRetriever
from profilerai_x.retrieval.fusion import reciprocal_rank_fusion
from profilerai_x.retrieval.metrics import evaluate

def main(n: int = 100, top_k: int = 10) -> None:
    rs = load_msmarco(n_queries=n)
    print(f"[load] corpus={len(rs.docs)} queries={len(rs.queries)}")

    t0 = time.perf_counter()
    bm = BM25().fit(rs.docs)
    print(f"[bm25]  fit {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    dense = DenseRetriever().fit(rs.docs)
    print(f"[dense] fit {time.perf_counter()-t0:.2f}s")

    bm_rankings, dense_rankings, hybrid_rankings = [], [], []
    t0 = time.perf_counter()
    for _, q in rs.queries:
        bm_r  = [d for d, _ in bm.search(q, top_k=50)]
        dn_r  = [d for d, _ in dense.search(q, top_k=50)]
        hy_r = [d for d, _ in reciprocal_rank_fusion([bm_r, dn_r], weights=[0.3, 0.7], top_k=top_k)]
        bm_rankings.append(bm_r[:top_k])
        dense_rankings.append(dn_r[:top_k])
        hybrid_rankings.append(hy_r)
    print(f"[search] {time.perf_counter()-t0:.2f}s total")

    gold = [rs.qrels[qid] for qid, _ in rs.queries]
    for name, r in [("BM25", bm_rankings), ("Dense", dense_rankings), ("Hybrid", hybrid_rankings)]:
        m = evaluate(r, gold, ks=(1, 5, 10))
        print(f"\n[{name}]")
        for k, v in m.items():
            print(f"  {k:>10s}: {v:.4f}" if k != "n_queries" else f"  {k:>10s}: {int(v)}")

if __name__ == "__main__":
    main()