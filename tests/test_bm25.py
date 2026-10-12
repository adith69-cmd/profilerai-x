from profilerai_x.retrieval.bm25 import BM25, tokenize

def test_tokenize():
    assert tokenize("Hello, World! 2024") == ["hello", "world", "2024"]

def test_idf_positive_and_decreasing():
    bm = BM25().fit(["a b", "a c", "a d", "b c"])
    assert bm.idf("a") > 0
    assert bm.idf("a") < bm.idf("d")

def test_rare_term_ranks_first():
    bm = BM25().fit(["cat dog", "cat fish", "fish bird"])
    assert bm.search("bird", top_k=1)[0][0] == 2

def test_unknown_term_scores_zero():
    bm = BM25().fit(["cat dog", "fish bird"])
    assert bm.score("zebra", 0) == 0.0