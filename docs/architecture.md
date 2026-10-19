# Architecture

Claims and wiki sentences enter through JSONL readers. Wiki pages are flattened without losing page title or sentence ID. The tokenizer applies Unicode NFKC normalization and case folding while retaining negation and numeric tokens. `InvertedIndex` stores evidence metadata, per-document token counts and term postings; it serializes to JSON for reload.

Retrievers: Boolean set operations; TF-IDF/cosine via scikit-learn; and BM25 directly over postings. The pipeline returns actual evidence with source metadata and makes a deliberately conservative NOT ENOUGH INFO decision because no evidence-pair verifier is trained. Streamlit loads a saved index and presents evidence and the limitations.
