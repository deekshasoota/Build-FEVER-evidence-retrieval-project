"""Robust JSON Lines loaders for FEVER claims and sentence knowledge bases."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)
VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

def read_jsonl(path: str | Path, *, strict: bool = False) -> Iterator[dict[str, Any]]:
    """Yield valid JSON objects, logging malformed lines unless strict=True."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("record must be a JSON object")
                yield row
            except (json.JSONDecodeError, ValueError) as exc:
                if strict:
                    raise ValueError(f"{path}:{line_no}: {exc}") from exc
                logger.warning("Skipping malformed record at %s:%d: %s", path, line_no, exc)

def load_claims(path: str | Path, *, strict: bool = False) -> list[dict[str, Any]]:
    """Load FEVER claim records; discard malformed/duplicate IDs and invalid labels."""
    claims, seen = [], set()
    for row in read_jsonl(path, strict=strict):
        cid = row.get("id")
        if cid is None or cid in seen or not isinstance(row.get("claim"), str):
            logger.warning("Skipping claim with missing/duplicate id or missing claim text")
            continue
        label = row.get("label")
        if label is not None and label not in VALID_LABELS:
            logger.warning("Skipping claim %s with invalid label %r", cid, label)
            continue
        seen.add(cid)
        claims.append(row)
    return claims

def iter_wiki_sentences(path: str | Path, *, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Stream and flatten FEVER wiki-pages JSONL without retaining all pages."""
    count = 0
    for page in read_jsonl(path):
        title = page.get("title") or page.get("id")
        raw_lines = page.get("lines", [])
        sentence_lines = raw_lines.splitlines() if isinstance(raw_lines, str) else raw_lines
        if not isinstance(sentence_lines, list):
            logger.warning("Skipping page %r with invalid lines field", title)
            continue
        for item in sentence_lines:
            try:
                sid, text = item.split("\t", 1)
                sid = int(sid)
            except (ValueError, AttributeError):
                logger.warning("Skipping malformed sentence on page %r", title)
                continue
            yield {"id": f"{title}::{sid}", "page": title, "sentence_id": sid, "text": text}
            count += 1
            if limit is not None and count >= limit:
                return

def load_wiki_sentences(path: str | Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    """Convenience list loader; use the iterator for large corpora."""
    return list(iter_wiki_sentences(path, limit=limit))
