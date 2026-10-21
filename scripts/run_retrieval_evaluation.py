"""Compare BM25, TF-IDF, sentence embeddings, and RRF on labeled claims."""
import argparse
import json
import time
from pathlib import Path

from src.data.load_data import load_claims
from src.evaluation.fever_evaluation import acceptable_evidence_sets
from src.evaluation.retrieval_metrics import (evidence_set_complete, ndcg_at_k,
    average_precision_at_k, precision_at_k, precision_recall_at_each_k,
    reciprocal_rank, recall_at_k)
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.bm25_retriever import search
from src.retrieval import embedding_retriever, rrf_retriever, tfidf_retriever

def _metric_summary(ranked_by_query: list[tuple[list[str], set[str], list[set[str]], float]], top_k: int) -> dict:
    if not ranked_by_query:
        return {"claims_with_evidence": 0, "metrics": {}, "mean_latency_ms": None, "curve": []}
    names = ("precision_at_10", "recall_at_10", "map_at_10", "ndcg_at_10", "mrr", "evidence_set_complete")
    totals = dict.fromkeys(names, 0.0)
    curve = [{"k": k, "precision": 0.0, "recall": 0.0} for k in range(1, top_k + 1)]
    total_latency = 0.0
    for ranked, relevant, groups, latency in ranked_by_query:
        total_latency += latency
        totals["precision_at_10"] += precision_at_k(ranked, relevant, min(10, top_k))
        totals["recall_at_10"] += recall_at_k(ranked, relevant, min(10, top_k))
        totals["map_at_10"] += average_precision_at_k(ranked, relevant, min(10, top_k))
        totals["ndcg_at_10"] += ndcg_at_k(ranked, relevant, min(10, top_k))
        totals["mrr"] += reciprocal_rank(ranked, relevant)
        totals["evidence_set_complete"] += float(evidence_set_complete(set(ranked), groups))
        for point in precision_recall_at_each_k(ranked, relevant, top_k):
            curve[point["k"] - 1]["precision"] += point["precision"]
            curve[point["k"] - 1]["recall"] += point["recall"]
    count = len(ranked_by_query)
    return {
        "claims_with_evidence": count,
        "metrics": {name: value / count for name, value in totals.items()},
        "mean_latency_ms": total_latency * 1000 / count,
        "curve": [{**point, "precision": point["precision"] / count,
                   "recall": point["recall"] / count} for point in curve],
    }


def evaluate_comparison(
    claims: list[dict], index: InvertedIndex, top_k: int = 10,
    domain_field: str = "domain", embedding_model_name: str = embedding_retriever.DEFAULT_MODEL,
) -> dict:
    """Evaluate all four required rankers overall and separately by domain."""
    records = list(index.documents.values())
    preprocessing = getattr(index, "preprocessing", {})
    encoder, vectors = embedding_retriever.encode_records(records, embedding_model_name)
    tfidf_vectorizer, tfidf_matrix = tfidf_retriever.build_index(records, preprocessing)
    model_names = ["BM25", "TF-IDF", "Sentence Embedding", "Hybrid RRF"]
    collected: dict[str, dict[str, list[tuple[list[str], set[str], list[set[str]], float]]]] = {}
    valid_claim_count = 0
    missing_domain_count = 0

    for claim in claims:
        groups = acceptable_evidence_sets(claim.get("evidence"))
        if not groups:
            continue
        valid_claim_count += 1
        raw_domain = claim.get(domain_field)
        if raw_domain is None:
            raw_domain = claim.get("domain") or claim.get("topic") or claim.get("category")
        if raw_domain is None or not str(raw_domain).strip():
            domain = "Unassigned"
            missing_domain_count += 1
        else:
            domain = str(raw_domain).strip()

        query = str(claim.get("claim", ""))
        started = time.perf_counter()
        bm25_rows = search(query, index, top_k)
        bm25_latency = time.perf_counter() - started

        started = time.perf_counter()
        tfidf_rows = tfidf_retriever.search_index(
            query, records, tfidf_vectorizer, tfidf_matrix, top_k, preprocessing,
        )
        tfidf_latency = time.perf_counter() - started

        started = time.perf_counter()
        embedding_rows = embedding_retriever.search(
            query, records, top_k, model=encoder, vectors=vectors,
        )
        embedding_latency = time.perf_counter() - started

        started = time.perf_counter()
        hybrid_rows = rrf_retriever.fuse([bm25_rows, tfidf_rows, embedding_rows], top_k)
        hybrid_latency = time.perf_counter() - started

        rows_by_model = [bm25_rows, tfidf_rows, embedding_rows, hybrid_rows]
        latencies = [bm25_latency, tfidf_latency, embedding_latency, hybrid_latency]
        relevant = set().union(*groups)
        bucket = collected.setdefault(domain, {name: [] for name in model_names})
        for name, rows, latency in zip(model_names, rows_by_model, latencies):
            bucket[name].append(([str(row["id"]) for row in rows], relevant, groups, latency))

    domains = {
        domain: {name: _metric_summary(values, top_k) for name, values in models.items()}
        for domain, models in sorted(collected.items())
    }
    all_by_model: dict[str, list[tuple[list[str], set[str], list[set[str]], float]]] = {
        name: [entry for domain in collected.values() for entry in domain[name]]
        for name in model_names
    }
    return {
        "split": "development",
        "top_k": top_k,
        "evidence_index_documents": len(records),
        "evidence_index_scope": "Scores apply to the evidence records loaded from the supplied index only.",
        "domain_field": domain_field,
        "claims_with_evidence": valid_claim_count,
        "claims_missing_domain": missing_domain_count,
        "domains_found": len([name for name in collected if name != "Unassigned"]),
        "models": model_names,
        "overall": {name: _metric_summary(values, top_k) for name, values in all_by_model.items()},
        "by_domain": domains,
        "embedding_model": embedding_model_name,
        "preprocessing": preprocessing,
        "note": "TF-IDF is fit over the full indexed corpus. Hybrid RRF fuses BM25, TF-IDF, and sentence-embedding rankings. Metrics use the union of acceptable evidence annotations; exact evidence completeness requires one full gold evidence group.",
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claims", required=True, help="Labeled FEVER split JSONL")
    parser.add_argument("--index", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--domain-field", default="domain", help="Claim JSONL field containing the domain label")
    parser.add_argument("--embedding-model", default=embedding_retriever.DEFAULT_MODEL)
    parser.add_argument("--output", default="artifacts/results/retriever_comparison.json")
    args = parser.parse_args()
    if args.top_k < 10:
        parser.error("--top-k must be at least 10 because the comparison reports @10 metrics")
    claims = load_claims(args.claims)
    if args.limit:
        claims = claims[:args.limit]
    report = evaluate_comparison(
        claims, InvertedIndex.load(args.index), args.top_k,
        domain_field=args.domain_field, embedding_model_name=args.embedding_model,
    )
    report.update({"split_file": Path(args.claims).name, "index_file": Path(args.index).name})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
