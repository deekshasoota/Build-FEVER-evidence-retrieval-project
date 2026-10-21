"""Disk-backed FEVER sentence index using SQLite FTS5 and BM25 ranking."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.preprocessing.text_preprocessor import DEFAULT_CONFIG, tokenize_with_config


class SQLiteEvidenceIndex:
    """Query a large evidence collection without loading it all into memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.metadata_path = self.path.with_suffix(".meta.json")
        self.metadata: dict[str, Any] = {}
        if self.metadata_path.exists():
            self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        with self._connect() as db:
            columns = {row[1] for row in db.execute("PRAGMA table_info(evidence_fts)")}
        self.has_preprocessed_terms = "terms" in columns
        count = self.metadata.get("sentences")
        if count is None:
            with self._connect() as db:
                count = db.execute("SELECT count(*) FROM evidence_fts").fetchone()[0]
        self._document_count = int(count)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, timeout=30)

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def term_count(self) -> int | None:
        value = self.metadata.get("terms")
        return int(value) if value is not None else None

    @property
    def preprocessing(self) -> dict[str, bool]:
        return {**DEFAULT_CONFIG, **self.metadata.get("preprocessing", {})}

    def tokenize(self, text: str) -> list[str]:
        return tokenize_with_config(text, self.preprocessing)

    @property
    def is_sample(self) -> bool:
        return str(self.metadata.get("source", "")).startswith("data/sample")

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        terms = self.tokenize(query)
        if not terms or top_k < 1:
            return []
        term_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        expression = f"terms : ({term_query})" if self.has_preprocessed_terms else term_query
        sql = """
            SELECT doc_id, page, sentence_id, text,
                   bm25(evidence_fts, 0.0, 0.0, 0.0, 1.0) AS score
            FROM evidence_fts
            WHERE evidence_fts MATCH ?
            ORDER BY score ASC
            LIMIT ?
        """
        with self._connect() as db:
            rows = db.execute(sql, (expression, top_k)).fetchall()
        return [
            {"id": doc_id, "page": page, "sentence_id": sentence_id, "text": text,
             "score": max(0.0, -float(score)), "rank": rank}
            for rank, (doc_id, page, sentence_id, text, score) in enumerate(rows, 1)
        ]

    def candidates(self, terms: list[str]) -> set[str]:
        if not terms:
            return set()
        term_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        expression = f"terms : ({term_query})" if self.has_preprocessed_terms else term_query
        with self._connect() as db:
            rows = db.execute("SELECT doc_id FROM evidence_fts WHERE evidence_fts MATCH ?", (expression,))
            return {row[0] for row in rows}

    def get_documents(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in doc_ids)
        with self._connect() as db:
            rows = db.execute(
                f"SELECT doc_id, page, sentence_id, text FROM evidence_fts WHERE doc_id IN ({placeholders})",
                doc_ids,
            ).fetchall()
        found = {row[0]: {"id": row[0], "page": row[1], "sentence_id": row[2], "text": row[3]} for row in rows}
        return [found[doc_id] for doc_id in doc_ids if doc_id in found]

    def all_documents(self) -> list[dict[str, Any]]:
        """Read the indexed evidence in a stable order for dense retrieval."""
        with self._connect() as db:
            rows = db.execute(
                "SELECT doc_id, page, sentence_id, text FROM evidence_fts ORDER BY doc_id"
            ).fetchall()
        return [
            {"id": row[0], "page": row[1], "sentence_id": row[2], "text": row[3]}
            for row in rows
        ]
