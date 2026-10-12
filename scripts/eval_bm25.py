"""Run BM25 on MS MARCO subset and print results."""
from __future__ import annotations
import time
from profilerai_x.data.loaders import load_msmarco
from profilerai_x.retrieval.bm25 import BM25
from profilerai_x.retrieval.metrics import evaluate

def main(n: int = 100, top_k: int = 10) -> None:
    print(f"[load] MS MARCO, n_queries={n}")
    rs = load_msmarco(n_queries=n)
    print(f"[load] corpus={len(rs.docs)} queries={len(rs.queries)}")

    t0 = time.perf_counter()
    bm = BM25().fit(rs.docs)
    fit_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rankings = [[d for d, _ in bm.search(q, top_k=top_k)] for _, q in rs.queries]
    search_s = time.perf_counter() - t0

    gold = [rs.qrels[qid] for qid, _ in rs.queries]
    m = evaluate(rankings, gold, ks=(1, 5, 10))

    print(f"[fit]    {fit_s:.2f}s  avgdl={bm.avgdl:.1f}")
    print(f"[search] {search_s:.2f}s  ({search_s/len(rs.queries)*1000:.1f} ms/query)")
    print("[results]")
    for k, v in m.items():
        print(f"  {k:>10s}: {v:.4f}" if k != "n_queries" else f"  {k:>10s}: {int(v)}")

if __name__ == "__main__":
    main()