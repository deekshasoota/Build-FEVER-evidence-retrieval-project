"""FEVER evidence-group parsing and retrieval/verdict evaluation helpers."""
from __future__ import annotations
from typing import Any

def acceptable_evidence_sets(raw: Any) -> list[set[str]]:
    """Convert FEVER annotator groups to page::sentence IDs; ignore malformed entries."""
    result: list[set[str]] = []
    if not isinstance(raw, list):
        return result
    for group in raw:
        if not isinstance(group, list):
            continue
        ids: set[str] = set()
        for item in group:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            page, sentence_id = item[2], item[3]
            if page is not None and sentence_id is not None:
                ids.add(f"{page}::{sentence_id}")
        if ids:
            result.append(ids)
    return result

def fever_score(gold_labels: list[str], predicted_labels: list[str], evidence_ok: list[bool]) -> float:
    """FEVER score: correct label plus complete evidence, except NEI needs no evidence."""
    if not gold_labels or len(gold_labels) != len(predicted_labels) or len(gold_labels) != len(evidence_ok):
        return 0.0
    correct = sum(1 for gold, pred, complete in zip(gold_labels, predicted_labels, evidence_ok)
                  if gold == pred and (gold == "NOT ENOUGH INFO" or complete))
    return correct / len(gold_labels)
