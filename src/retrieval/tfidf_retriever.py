"""Corpus-wide TF-IDF and cosine retrieval, using scikit-learn."""
from src.preprocessing.text_preprocessor import tokenize, tokenize_with_config

def build_index(records: list[dict], preprocessing: dict | None = None):
    """Fit one TF-IDF vocabulary and IDF distribution over the full corpus."""
    tokenizer = (lambda text: tokenize_with_config(text, preprocessing)) if preprocessing else tokenize
    if not records or not any(tokenizer(str(row.get("text", ""))) for row in records):
        return None, None
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [str(row.get("text", "")) for row in records]
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, lowercase=False)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def search_index(query: str, records: list[dict], vectorizer, matrix, top_k: int = 10,
                 preprocessing: dict | None = None) -> list[dict]:
    """Rank against a previously fitted corpus-wide TF-IDF matrix."""
    query_tokens = tokenize_with_config(query, preprocessing) if preprocessing else tokenize(query)
    if not records or top_k < 1 or vectorizer is None or matrix is None or not query_tokens:
        return []
    from sklearn.metrics.pairwise import cosine_similarity
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = sorted(range(len(records)), key=lambda i: (-scores[i], str(records[i].get("id", ""))))
    return [{**records[i], "score": float(scores[i]), "rank": rank}
            for rank, i in enumerate([i for i in ranked if scores[i] > 0][:top_k], 1)]

def search(query: str, records: list[dict], top_k: int = 5, preprocessing: dict | None = None) -> list[dict]:
    vectorizer, matrix = build_index(records, preprocessing)
    return search_index(query, records, vectorizer, matrix, top_k, preprocessing)
