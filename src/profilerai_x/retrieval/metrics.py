"""Hit@K, MRR, NDCG@K. Binary relevance."""
from __future__ import annotations
import math

def hit_at_k(ranked: list[int], rel: set[int], k: int) -> float:
    return 1.0 if any(d in rel for d in ranked[:k]) else 0.0

def mrr(ranked: list[int], rel: set[int]) -> float:
    for r, d in enumerate(ranked, 1):
        if d in rel:
            return 1.0 / r
    return 0.0

def ndcg_at_k(ranked: list[int], rel: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(r + 1) for r, d in enumerate(ranked[:k], 1) if d in rel)
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(rel), k) + 1))
    return dcg / idcg if idcg else 0.0

def evaluate(rankings: list[list[int]], qrels: list[set[int]], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    valid = [(r, q) for r, q in zip(rankings, qrels) if q]
    out: dict[str, float] = {}
    if not valid:
        for k in ks:
            out[f"hit@{k}"] = 0.0
        out["mrr"] = 0.0
        for k in ks:
            out[f"ndcg@{k}"] = 0.0
        out["n_queries"] = 0.0
        return out
    for k in ks:
        out[f"hit@{k}"] = sum(hit_at_k(r, q, k) for r, q in valid) / len(valid)
    out["mrr"] = sum(mrr(r, q) for r, q in valid) / len(valid)
    for k in ks:
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, q, k) for r, q in valid) / len(valid)
    out["n_queries"] = float(len(valid))
    return out