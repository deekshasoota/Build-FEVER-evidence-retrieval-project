"""Dataset summary helpers."""
from collections import Counter
from typing import Any

def summarize_claims(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"records": len(rows), "labels": dict(Counter(r.get("label", "UNLABELED") for r in rows)),
            "missing_claims": sum(not r.get("claim") for r in rows),
            "with_evidence": sum(bool(r.get("evidence")) for r in rows)}
