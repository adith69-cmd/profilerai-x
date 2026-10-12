from profilerai_x.retrieval.metrics import hit_at_k, mrr, ndcg_at_k

def test_hit():
    assert hit_at_k([5, 7, 1], {1}, 3) == 1.0
    assert hit_at_k([5, 7, 1], {1}, 2) == 0.0

def test_mrr():
    assert mrr([1, 2, 3], {3}) == 1/3
    assert mrr([1, 2, 3], {1}) == 1.0

def test_ndcg_perfect_ranking():
    assert ndcg_at_k([1, 2, 3], {1, 2}, 2) == 1.0

def test_ndcg_relevant_later_is_worse():
    assert ndcg_at_k([1, 2, 3], {1}, 3) > ndcg_at_k([1, 2, 3], {3}, 3)