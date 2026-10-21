"""Dense semantic retrieval with a Sentence Transformers encoder."""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(model_name: str = DEFAULT_MODEL):
    """Load the optional encoder only when semantic retrieval is requested."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence embedding retrieval needs sentence-transformers. "
            "Install the project requirements and restart the app."
        ) from exc
    return SentenceTransformer(model_name)


def encode_records(records: list[dict[str, Any]], model_name: str = DEFAULT_MODEL):
    """Return an encoder and normalized vectors in the same order as records."""
    model = load_model(model_name)
    texts = [str(record.get("text", "")) for record in records]
    vectors = model.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False, batch_size=64,
    )
    return model, np.asarray(vectors, dtype=np.float32)


def search(
    query: str,
    records: list[dict[str, Any]],
    top_k: int = 10,
    *,
    model=None,
    vectors: np.ndarray | None = None,
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, Any]]:
    """Rank records by cosine similarity to the query embedding."""
    if top_k < 1 or not records or not query.strip():
        return []
    if model is None:
        model = load_model(model_name)
    if vectors is None:
        vectors = np.asarray(model.encode(
            [str(record.get("text", "")) for record in records],
            convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False, batch_size=64,
        ), dtype=np.float32)
    query_vector = np.asarray(model.encode(
        [query], convert_to_numpy=True, normalize_embeddings=True,
        show_progress_bar=False,
    )[0], dtype=np.float32)
    scores = np.asarray(vectors) @ query_vector
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        {**records[int(position)], "score": float(scores[position]), "rank": rank}
        for rank, position in enumerate(order, start=1)
    ]
