# BM25 from Scratch: Why I Wrote a Search Engine in 60 Lines

Every RAG tutorial starts the same way: `pip install rank-bm25`, import,
call `get_scores()`, done. That's fine if you're shipping a product and
need to move on. It's a trap if you're trying to understand retrieval
deeply enough to defend it in a technical interview.

So I wrote BM25 from scratch. Here's what I actually learned.

## What BM25 does in one paragraph

Given a query, BM25 scores every document in a corpus by how well its
terms match the query terms. But three things make it more than "count
the overlapping words":

1. **Rare terms matter more than common ones.** A query containing
   "photosynthesis" should weight that term more than "the". This is
   Inverse Document Frequency.
2. **Long documents shouldn't win by default.** A 2000-word document
   contains more words, so it will match more query terms by chance.
   BM25 normalizes by document length.
3. **Repeating a term has diminishing returns.** A document that says
   "python" 50 times is not 50× more relevant to a query for "python"
   than a document that says it once. This is term saturation, controlled
   by the parameter k1.

## The formula

The score of document D for query Q is the sum over query terms:

    score(D, Q) = Σ_{t ∈ Q} IDF(t) · [ f(t, D) · (k1 + 1) ]
                            ─────────────────────────────────
                            f(t, D) + k1 · (1 - b + b · |D|/avgdl)

Where:
- f(t, D)  = how many times term t appears in document D
- |D|      = length of D in tokens
- avgdl    = average document length in the corpus
- k1 = 1.5, b = 0.75 (standard defaults from the TREC-era literature)

The IDF term uses the smoothed form `log((N - df + 0.5)/(df + 0.5) + 1)`,
which stays positive even for terms that appear in most documents.

## My implementation

    [Paste bm25.py here]

The only non-obvious part is the length-normalization term:

    norm = k1 * (1 - b + b * dl / avgdl)

When `b = 1`, documents are fully normalized by length — a 100-token
document and a 1000-token document with the same term frequency end up
scoring differently. When `b = 0`, no normalization happens at all. The
default `b = 0.75` sits in between.

## Evaluation on MS MARCO

I evaluated on a 100-query subset of the MS MARCO v1.1 validation set.
The corpus is the union of the passages those 100 queries retrieved —
821 passages total. This is not comparable to published MS MARCO
leaderboard numbers, since those use the full 8.8M-passage corpus.

    [Paste S1 results table here]

Hit@10 was 0.928 — the correct passage was in the top 10 for 93% of
queries. But MRR was only 0.435, which means the correct passage was
rarely ranked first.

## The moment my metric looked wrong

I ran a diagnostic on the first query: `"walgreens store sales average"`.

    Q: walgreens store sales average
    Relevant: {0}

    1. score=14.128 [irrelevant] ...$4 billion in sales in 1987...
    2. score=12.930 [irrelevant] ...Your Walgreens Store...
    3. score=12.175 [irrelevant] ...Walgreens opened 184 new locations...
    4. score=9.011  [RELEVANT]   ...The average Walgreens salary ranges...

The "relevant" passage per the qrel is about Walgreens **salaries**, not
sales. My top-3 results are about Walgreens **store sales, counts, and
openings** — arguably more relevant to the query than the qrel itself.

The cause: MS MARCO qrels are single-annotator clicks. A human clicked
one passage once, and that's the "ground truth." On a 10-passage subset,
the non-relevant passages are often topically near-identical to the
relevant one. My BM25 is not broken. The benchmark's labels are noisy.

I chose not to tune BM25 to chase this. Instead I documented it and
moved to the techniques that actually improve ranking: dense retrieval
and cross-encoder reranking.

## What came next

Two sessions later, the pipeline looks like this:

    BM25 (S1)                   MRR 0.435
    + Dense (S2)                MRR 0.573
    + Cross-encoder rerank (S3) MRR 0.679

Same corpus, same queries, same evaluation harness. The full results
and failure analysis are in the repo:

https://github.com/adith69-cmd/profilerai-x

## What I'd tell someone starting this

1. Write BM25 yourself. It takes an hour and you'll understand retrieval
   in a way that library calls can't teach.
2. Evaluate on a real benchmark early. The Walgreens bug would have been
   invisible without the metrics.
3. When a number looks wrong, investigate the *data* before you
   investigate the *code*. Half the time it's the labels.

## Repo

https://github.com/adith69-cmd/profilerai-x