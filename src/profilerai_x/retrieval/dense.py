"""Dense retrieval with Sentence-Transformers + cosine similarity."""
from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class DenseRetriever:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self._emb: np.ndarray | None = None

    def fit(self, corpus: list[str]) -> "DenseRetriever":
        self._emb = self.model.encode(
            corpus, batch_size=32, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True,
        )
        return self

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        scores = self._emb @ q                       # cosine sim (both normalized)
        order = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in order]