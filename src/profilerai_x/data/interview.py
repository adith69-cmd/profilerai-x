"""Load the Shlok307/Interview_questions dataset as a RetrievalSet.

Task: given a question, retrieve the correct answer from a pool of 5002
answers. qrel for query i = index of answer i. Same shape as MS MARCO
(query -> passage) but different domain and much harder negatives,
since many AI/programming answers share vocabulary.
"""
from __future__ import annotations
from datasets import load_dataset
from profilerai_x.data.loaders import RetrievalSet


def load_interview(n_queries: int = 500) -> RetrievalSet:
    ds = load_dataset("Shlok307/Interview_questions", split="train")

    docs: list[str] = []
    for row in ds:
        docs.append(f"Q: {row['question']}\nA: {row['answer']}")

    n_q = min(n_queries, len(ds))
    queries: list[tuple[int, str]] = []
    qrels: dict[int, set[int]] = {}
    for i in range(n_q):
        queries.append((i, ds[i]["question"]))
        qrels[i] = {i}

    return RetrievalSet(docs=docs, queries=queries, qrels=qrels)