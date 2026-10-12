"""MS MARCO v1.1 subset loader. NOT comparable to full leaderboard."""
from __future__ import annotations
from dataclasses import dataclass
from datasets import load_dataset

@dataclass
class RetrievalSet:
    docs: list[str]
    queries: list[tuple[int, str]]
    qrels: dict[int, set[int]]

def load_msmarco(n_queries: int = 100) -> RetrievalSet:
    ds = load_dataset("microsoft/ms_marco", "v1.1", split=f"validation[:{n_queries}]")
    docs: list[str] = []
    doc_id: dict[str, int] = {}
    queries: list[tuple[int, str]] = []
    qrels: dict[int, set[int]] = {}

    for row in ds:
        qid = int(row["query_id"])
        queries.append((qid, row["query"]))
        rel: set[int] = set()
        for text, sel in zip(row["passages"]["passage_text"], row["passages"]["is_selected"]):
            if text not in doc_id:
                doc_id[text] = len(docs)
                docs.append(text)
            if int(sel) == 1:
                rel.add(doc_id[text])
        qrels[qid] = rel

    return RetrievalSet(docs=docs, queries=queries, qrels=qrels)