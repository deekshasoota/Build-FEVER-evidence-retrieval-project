"""Sentence relevance metrics and evidence-set completeness."""
import math

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(x in relevant for x in retrieved[:k]) / k if k > 0 else 0.0

def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(x in relevant for x in retrieved[:k]) / len(relevant) if relevant else 0.0

def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    return next((1 / i for i, item in enumerate(retrieved, 1) if item in relevant), 0.0)

def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum((1 / math.log2(i + 2)) for i, item in enumerate(retrieved[:k]) if item in relevant)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal else 0.0

def evidence_set_complete(retrieved: set[str], acceptable_sets: list[set[str]]) -> bool:
    return any(gold <= retrieved for gold in acceptable_sets if gold)
