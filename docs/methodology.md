# Methodology

## Retrieval

For a query term with document frequency `df` in a collection of `N` records, BM25 uses `log(1 + (N - df + 0.5)/(df + 0.5))` as its IDF factor. Term frequency is saturated with `k1`; document length is normalized by average length and parameter `b`. Defaults are `k1=1.5`, `b=0.75`. TF-IDF/cosine is fitted over the loaded corpus. Sentence embeddings use MiniLM cosine similarity. Hybrid RRF combines the three ranked lists using `1/(60 + rank)` per retriever. Boolean retrieval provides AND/OR/NOT set operations and has no ranking score.

## Evidence and verification

Evidence records identify page and sentence. Retrieval relevance is not a truth label. The baseline logistic regression uses six aggregate features from top-ranked evidence: maximum claim-token coverage, maximum token Jaccard, negation agreement, number agreement, evidence count, and scaled maximum BM25. It trains on training labels and retrieved text; the development split is evaluated separately. The model is intentionally simple and does not fully reason over multiple evidence sets or resolve conflicting evidence. Without a saved model, inference explicitly abstains.

## Preprocessing choices

The implementation uses NFKC, case folding and a regex tokenizer. Stop-word removal and NLTK Porter stemming are optional build-time settings, saved with the index so the same processing is applied to lexical queries. Stop-word filtering retains negation. Original text and evidence identifiers remain unchanged for display and scoring. Build a fresh index for every preprocessing configuration and compare variants on development evidence recall before selecting one.
