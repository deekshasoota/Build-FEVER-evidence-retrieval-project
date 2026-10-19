"""Train a reproducible logistic-regression verifier from retrieved train evidence."""
from __future__ import annotations
import json, logging, pickle
from pathlib import Path
from typing import Any
from src.data.load_data import load_claims
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retriever import search
from src.verification.feature_extractor import extract_features

logger = logging.getLogger(__name__)

def create_training_rows(claims: list[dict[str, Any]], index: InvertedIndex, top_k: int = 10):
    x, y = [], []
    for n, claim in enumerate(claims, 1):
        label = claim.get("label")
        if label not in {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}: continue
        retrieved = search(claim["claim"], index, top_k)
        x.append(extract_features(claim["claim"], retrieved)); y.append(label)
        if n % 1000 == 0: logger.info("Processed %d claims", n)
    return x, y

def train(train_path: str, index_path: str, model_path: str, *, limit: int | None = None, seed: int = 42) -> dict:
    from sklearn.linear_model import LogisticRegression
    claims = load_claims(train_path)
    if limit: claims = claims[:limit]
    index = InvertedIndex.load(index_path)
    x, y = create_training_rows(claims, index)
    if len(set(y)) < 2: raise ValueError("Training requires at least two represented labels; supply the official labeled train split.")
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(x, y)
    path = Path(model_path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f: pickle.dump(model, f)
    metadata = {"algorithm": "scikit-learn LogisticRegression", "seed": seed, "records": len(y),
                "label_counts": {label: y.count(label) for label in sorted(set(y))}, "features": ["claim coverage", "token Jaccard", "negation match", "number match", "evidence count", "scaled maximum BM25"]}
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
