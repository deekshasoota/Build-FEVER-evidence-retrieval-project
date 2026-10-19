# Methodology

## Retrieval

For a query term with document frequency `df` in a collection of `N` records, BM25 uses `log(1 + (N - df + 0.5)/(df + 0.5))` as its IDF factor. Term frequency is saturated with `k1`; document length is normalized by average length and parameter `b`. Defaults are `k1=1.5`, `b=0.75`. Terms that occur in no postings return no candidates. TF-IDF/cosine is supplied as a comparison method. Boolean retrieval provides AND/OR/NOT set operations and has no ranking score.

## Evidence and verification

Evidence records identify page and sentence. Retrieval relevance is not a truth label. The baseline logistic regression uses six aggregate features from top-ranked evidence: maximum claim-token coverage, maximum token Jaccard, negation agreement, number agreement, evidence count, and scaled maximum BM25. It trains on training labels and retrieved text; the development split is evaluated separately. The model is intentionally simple and does not fully reason over multiple evidence sets or resolve conflicting evidence. Without a saved model, inference explicitly abstains.

## Preprocessing choices

The implementation uses NFKC, case folding and a regex tokenizer. Stop words are off by default. The optional list excludes negation. Stemming/lemmatization are not enabled. Preserve original text and identifiers for display and evaluation. Evaluate preprocessing variants on development evidence recall before selecting one.
