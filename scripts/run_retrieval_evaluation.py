"""Evaluate BM25 sentence retrieval against FEVER evidence annotations."""
import argparse
import json
import time
from pathlib import Path

from src.data.load_data import load_claims
from src.evaluation.fever_evaluation import acceptable_evidence_sets
from src.evaluation.retrieval_metrics import (evidence_set_complete, ndcg_at_k,
    precision_at_k, reciprocal_rank, recall_at_k)
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retriever import search

def evaluate(claims: list[dict], index: InvertedIndex, top_k: int) -> dict:
    totals = {"precision_at_k": 0.0, "recall_at_k": 0.0, "mrr": 0.0,
              "ndcg_at_k": 0.0, "evidence_set_complete": 0.0, "latency_seconds": 0.0}
    count = 0
    for claim in claims:
        groups = acceptable_evidence_sets(claim.get("evidence"))
        if not groups:
            continue
        start = time.perf_counter()
        rows = search(claim.get("claim", ""), index, top_k)
        totals["latency_seconds"] += time.perf_counter() - start
        ranked = [str(row["id"]) for row in rows]
        relevant = set().union(*groups)
        totals["precision_at_k"] += precision_at_k(ranked, relevant, top_k)
        totals["recall_at_k"] += recall_at_k(ranked, relevant, top_k)
        totals["mrr"] += reciprocal_rank(ranked, relevant)
        totals["ndcg_at_k"] += ndcg_at_k(ranked, relevant, top_k)
        totals["evidence_set_complete"] += float(evidence_set_complete(set(ranked), groups))
        count += 1
    if count:
        for key in totals:
            totals[key] /= count
    return {"claims_with_evidence": count, "top_k": top_k,
            "mean_latency_ms": totals.pop("latency_seconds") * 1000,
            "metrics": totals,
            "note": "Sentence metrics use the union of acceptable evidence. Completeness requires all sentences from at least one acceptable gold group."}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, help="Labeled FEVER split JSONL")
    parser.add_argument("--index", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", default="artifacts/results/retrieval_metrics.json")
    args = parser.parse_args()
    claims = load_claims(args.claims)
    if args.limit:
        claims = claims[:args.limit]
    report = evaluate(claims, InvertedIndex.load(args.index), args.top_k)
    report.update({"split_file": str(args.claims), "retriever": "BM25"})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
