"""Interpretable claim/evidence aggregate features for a lightweight classifier."""
import re
from src.preprocessing.text_preprocessor import tokenize

NEGATIONS = {"not", "no", "never", "neither", "nor", "n't"}
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")

def extract_features(claim: str, evidence: list[dict]) -> list[float]:
    claim_terms = set(tokenize(claim))
    claim_numbers = set(NUMBER_RE.findall(claim))
    claim_neg = bool(claim_terms & NEGATIONS)
    overlaps, jaccards, neg_matches, number_matches, scores = [], [], [], [], []
    for item in evidence:
        terms = set(tokenize(str(item.get("text", ""))))
        overlap = len(claim_terms & terms)
        overlaps.append(overlap / max(1, len(claim_terms)))
        jaccards.append(overlap / max(1, len(claim_terms | terms)))
        neg_matches.append(float(claim_neg == bool(terms & NEGATIONS)))
        nums = set(NUMBER_RE.findall(str(item.get("text", ""))))
        number_matches.append(float(not claim_numbers or bool(claim_numbers & nums)))
        scores.append(float(item.get("score", 0.0)))
    return [max(overlaps, default=0), max(jaccards, default=0), max(neg_matches, default=0),
            max(number_matches, default=0), min(len(evidence), 20) / 20,
            max(scores, default=0) / (1 + max(scores, default=0))]
