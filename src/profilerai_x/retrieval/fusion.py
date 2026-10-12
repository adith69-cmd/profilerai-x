"""Reciprocal Rank Fusion. From scratch, ~15 lines.

RRF score(doc) = sum over retrievers of 1 / (k + rank_of_doc)
k=60 is the standard default from Cormack et al. 2009.
"""
from __future__ import annotations

def reciprocal_rank_fusion(
    rankings: list[list[int]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[int, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)
    ordered = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ordered[:top_k]