"""Transparent retrieval-first inference with an optional trained verifier."""
import pickle
from pathlib import Path
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retriever import search
from src.verification.feature_extractor import extract_features

def verify(claim: str, index: InvertedIndex, top_k: int = 5) -> dict:
    evidence = search(claim, index, top_k)
    if not evidence:
        return {"verdict": "NOT ENOUGH INFO", "method": "retrieval-only abstention", "evidence": [],
                "explanation": "No matching evidence was retrieved from the indexed knowledge base."}
    model_path = Path("artifacts/models/verifier.pkl")
    if model_path.exists():
        with model_path.open("rb") as f: model = pickle.load(f)
        features = extract_features(claim, evidence)
        verdict = str(model.predict([features])[0])
        return {"verdict": verdict, "method": "trained LogisticRegression over BM25 evidence features", "evidence": evidence,
                "explanation": "The verdict was predicted from interpretable overlap, negation, number, evidence-count, and retrieval-score features. It is a statistical prediction, not a proof; review the cited sentences."}
    return {"verdict": "NOT ENOUGH INFO", "method": "retrieval-only baseline", "evidence": evidence,
            "explanation": "Relevant-looking sentences were retrieved by BM25, but lexical relevance alone cannot establish entailment or contradiction. The system abstains until a verifier is trained on the official training split."}
