"""TF-IDF and cosine retrieval, using scikit-learn when installed."""
from src.preprocessing.text_preprocessor import tokenize

def search(query: str, records: list[dict], top_k: int = 5) -> list[dict]:
    if not records or top_k < 1 or not tokenize(query): return []
    if not any(tokenize(str(r.get("text", ""))) for r in records): return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [str(r.get("text", "")) for r in records]
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False)
    matrix = vectorizer.fit_transform(texts + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    ranked = sorted(range(len(records)), key=lambda i: (-scores[i], str(records[i].get("id", ""))))
    return [{**records[i], "score": float(scores[i]), "rank": rank} for rank, i in enumerate([i for i in ranked if scores[i] > 0][:top_k], 1)]
