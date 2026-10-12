"""Cross-encoder reranking."""
from __future__ import annotations
from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class CrossEncoderReranker:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[int],
        corpus: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        if not candidates:
            return []
        pairs = [(query, corpus[i]) for i in candidates]
        scores = self.model.predict(pairs, show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
        return [(int(i), float(s)) for i, s in ranked[:top_k]]