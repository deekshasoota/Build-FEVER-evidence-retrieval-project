"""Okapi BM25 over the inverted index."""
import math
from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessor import tokenize

def search(query: str, index: InvertedIndex, top_k: int = 5, k1: float = 1.5, b: float = .75) -> list[dict]:
    if top_k < 1: return []
    terms = tokenize(query); n = len(index.documents)
    if not terms or not n: return []
    avgdl = sum(index.lengths.values()) / n if n else 0
    scores: dict[str, float] = {}
    for term in terms:
        posting = index.postings.get(term, {})
        df = len(posting)
        if not df: continue
        idf = math.log(1 + (n - df + .5) / (df + .5))
        for doc_id, tf in posting.items():
            dl = index.lengths[doc_id]
            denom = tf + k1 * (1 - b + b * dl / avgdl) if avgdl else tf + k1
            scores[doc_id] = scores.get(doc_id, 0) + idf * tf * (k1 + 1) / denom
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
    return [{**index.documents[did], "score": score, "rank": rank} for rank, (did, score) in enumerate(ranked, 1)]
