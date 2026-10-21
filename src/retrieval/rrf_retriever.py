"""Reciprocal Rank Fusion for combining independently ranked result lists."""
from __future__ import annotations

from typing import Any


def fuse(rankings: list[list[dict[str, Any]]], top_k: int = 10, rank_constant: int = 60) -> list[dict[str, Any]]:
    """Fuse ranked records by ID using sum(1 / (rank_constant + rank))."""
    if top_k < 1:
        return []
    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for ranking in rankings:
        seen: set[str] = set()
        for rank, row in enumerate(ranking, start=1):
            doc_id = str(row.get("id", ""))
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rank_constant + rank)
            records.setdefault(doc_id, row)
    ordered = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
    return [
        {**records[doc_id], "score": scores[doc_id], "rank": rank}
        for rank, doc_id in enumerate(ordered, start=1)
    ]
