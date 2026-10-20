"""Memory-conscious positional inverted index with JSON persistence."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any
from src.preprocessing.text_preprocessor import tokenize

class InvertedIndex:
    def __init__(self) -> None:
        self.documents: dict[str, dict[str, Any]] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.lengths: dict[str, int] = {}

    @property
    def document_count(self) -> int:
        return len(self.documents)

    @property
    def term_count(self) -> int:
        return len(self.postings)

    @property
    def is_sample(self) -> bool:
        return bool(self.documents) and all(
            str(doc.get("page", "")).startswith("Example_") for doc in self.documents.values()
        )

    def get_documents(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]

    def add(self, record: dict[str, Any]) -> None:
        doc_id = str(record.get("id", ""))
        text = record.get("text", "")
        if not doc_id or not isinstance(text, str):
            raise ValueError("Each evidence record requires a non-empty id and string text")
        if doc_id in self.documents:
            raise ValueError(f"Duplicate evidence id: {doc_id}")
        terms = tokenize(text)
        self.documents[doc_id] = {**record, "text": text}
        self.lengths[doc_id] = len(terms)
        for term, freq in Counter(terms).items():
            self.postings[term][doc_id] = freq

    def candidates(self, terms: list[str]) -> set[str]:
        return {doc_id for term in terms for doc_id in self.postings.get(term, {})}

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self.documents, "postings": dict(self.postings), "lengths": self.lengths}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "InvertedIndex":
        obj = cls(); data = json.loads(Path(path).read_text(encoding="utf-8"))
        obj.documents = data["documents"]
        obj.postings = defaultdict(dict, data["postings"])
        obj.lengths = data["lengths"]
        return obj
