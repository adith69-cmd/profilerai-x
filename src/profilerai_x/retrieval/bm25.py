"""BM25 (Okapi) from scratch. k1=1.5, b=0.75."""
from __future__ import annotations
import math, re
from collections import Counter
from typing import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs: list[list[str]] = []
        self._tf: list[Counter[str]] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0

    def fit(self, corpus: Iterable[str]) -> "BM25":
        self._docs = [tokenize(d) for d in corpus]
        self._tf = [Counter(d) for d in self._docs]
        n = len(self._docs)
        df = Counter()
        for d in self._docs:
            df.update(set(d))
        self._avgdl = sum(len(d) for d in self._docs) / n if n else 0.0
        self._idf = {t: math.log((n - c + 0.5) / (c + 0.5) + 1.0) for t, c in df.items()}
        return self

    def score(self, query: str, i: int) -> float:
        dl = len(self._docs[i])
        tf = self._tf[i]
        norm = self.k1 * (1.0 - self.b + self.b * (dl / self._avgdl if self._avgdl else 0.0))
        s = 0.0
        for t in tokenize(query):
            f = tf.get(t, 0)
            if f == 0 or t not in self._idf:
                continue
            s += self._idf[t] * (f * (self.k1 + 1.0)) / (f + norm)
        return s

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scored = [(i, self.score(query, i)) for i in range(len(self._docs))]
        scored.sort(key=lambda x: (-x[1], x[0]))
        return scored[:top_k]

    @property
    def avgdl(self) -> float: return self._avgdl

    def idf(self, term: str) -> float: return self._idf.get(term, 0.0)