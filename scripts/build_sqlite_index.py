"""Build a disk-backed full-text index from one FEVER JSONL file or a directory."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from pathlib import Path

from src.data.load_data import iter_wiki_sentences
from src.preprocessing.text_preprocessor import tokenize_with_config
from src.utils.logging_config import configure_logging


def build(source: Path, output: Path, limit: int | None = None,
          preprocessing: dict[str, bool] | None = None) -> dict:
    preprocessing = {"remove_stopwords": False, "stemming": False, **(preprocessing or {})}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".building")
    if temporary.exists():
        temporary.unlink()
    started = time.perf_counter()
    db = sqlite3.connect(temporary)
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA cache_size=-65536")
        db.execute("CREATE VIRTUAL TABLE evidence_fts USING fts5(doc_id UNINDEXED, page UNINDEXED, sentence_id UNINDEXED, text UNINDEXED, terms, tokenize='unicode61')")
        batch = []
        inserted = 0
        for record in iter_wiki_sentences(source, limit=limit):
            terms = " ".join(tokenize_with_config(record["text"], preprocessing))
            batch.append((record["id"], str(record["page"]), int(record["sentence_id"]), record["text"], terms))
            if len(batch) >= 5000:
                db.executemany("INSERT INTO evidence_fts(doc_id,page,sentence_id,text,terms) VALUES(?,?,?,?,?)", batch)
                inserted += len(batch)
                db.commit()
                batch.clear()
                if inserted % 100000 == 0:
                    print(f"Indexed {inserted:,} sentences…", flush=True)
        if batch:
            db.executemany("INSERT INTO evidence_fts(doc_id,page,sentence_id,text,terms) VALUES(?,?,?,?,?)", batch)
            inserted += len(batch)
            db.commit()
        db.execute("INSERT INTO evidence_fts(evidence_fts) VALUES('optimize')")
        db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        db.close()
    os.replace(temporary, output)
    elapsed = time.perf_counter() - started
    metadata = {
        "sentences": inserted, "terms": None, "build_seconds": elapsed,
        "index_bytes": output.stat().st_size, "source": str(source),
        "limit": limit, "backend": "SQLite FTS5 / BM25", "preprocessing": preprocessing,
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wiki_jsonl", type=Path, help="FEVER JSONL file or extracted directory of JSONL files")
    parser.add_argument("--output", type=Path, default=Path("artifacts/indexes/fever_index.sqlite3"))
    parser.add_argument("--limit", type=int, help="Optional sentence cap for a preview index")
    parser.add_argument("--remove-stopwords", action="store_true", help="Remove English stop words while retaining negation")
    parser.add_argument("--stemming", action="store_true", help="Apply Porter stemming to indexed and query tokens")
    args = parser.parse_args()
    configure_logging()
    config = {"remove_stopwords": args.remove_stopwords, "stemming": args.stemming}
    print(json.dumps(build(args.wiki_jsonl, args.output, args.limit, config), indent=2), flush=True)


if __name__ == "__main__":
    main()
