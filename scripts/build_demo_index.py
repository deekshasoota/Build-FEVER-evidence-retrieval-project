"""Create a compact, deterministic deployment preview from a full FEVER FTS5 index."""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
from pathlib import Path

from src.preprocessing.text_preprocessor import tokenize

DEMO_QUERIES = (
    "Eiffel Tower Paris",
    "Pacific Ocean largest",
    "Barack Obama Hawaii",
    "Larry Bird basketball",
    "India capital New Delhi",
    "Taj Mahal India",
    "Mount Everest highest mountain",
    "Albert Einstein physicist",
    "World War II",
    "NASA space",
    "Google search engine",
    "cricket sport",
)


def build_demo_index(source: Path, output: Path, size: int = 100_000, seed: int = 2026) -> dict:
    if not source.is_file():
        raise FileNotFoundError(f"Source SQLite index does not exist: {source}")
    if source.resolve() == output.resolve():
        raise ValueError("The demo index output must be different from the full source index.")

    source_db = sqlite3.connect(f"file:{source.resolve()}?mode=ro", uri=True, timeout=60)
    try:
        total = int(source_db.execute("SELECT count(*) FROM evidence_fts").fetchone()[0])
        maximum = int(source_db.execute("SELECT max(rowid) FROM evidence_fts").fetchone()[0] or 0)
        if total == 0:
            raise ValueError("The source index contains no sentences.")
        sample_size = min(size, total)
        rowids = random.Random(seed).sample(range(1, maximum + 1), sample_size)

        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".building")
        temporary.unlink(missing_ok=True)
        demo_db = sqlite3.connect(temporary)
        try:
            demo_db.execute(
                "CREATE VIRTUAL TABLE evidence_fts USING "
                "fts5(doc_id UNINDEXED, page UNINDEXED, sentence_id UNINDEXED, text, tokenize='unicode61')"
            )
            inserted = 0
            seen_doc_ids: set[str] = set()
            for offset in range(0, len(rowids), 800):
                batch_ids = rowids[offset:offset + 800]
                query = (
                    "SELECT doc_id, page, sentence_id, text FROM evidence_fts "
                    f"WHERE rowid IN ({','.join('?' for _ in batch_ids)})"
                )
                rows = source_db.execute(query, batch_ids).fetchall()
                new_rows = [row for row in rows if row[0] not in seen_doc_ids]
                if new_rows:
                    demo_db.executemany(
                        "INSERT INTO evidence_fts(doc_id,page,sentence_id,text) VALUES(?,?,?,?)", new_rows
                    )
                    seen_doc_ids.update(row[0] for row in new_rows)
                    inserted += len(new_rows)
                if offset and offset % 20_000 == 0:
                    demo_db.commit()
                    print(f"Copied {inserted:,} sentences…", flush=True)

            for query_text in DEMO_QUERIES:
                terms = tokenize(query_text, remove_stopwords=True)
                expression = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
                query_rows = source_db.execute(
                    "SELECT doc_id, page, sentence_id, text FROM evidence_fts "
                    "WHERE evidence_fts MATCH ? ORDER BY bm25(evidence_fts,0,0,0,1), rowid LIMIT 12",
                    (expression,),
                ).fetchall()
                new_rows = [row for row in query_rows if row[0] not in seen_doc_ids]
                if new_rows:
                    demo_db.executemany(
                        "INSERT INTO evidence_fts(doc_id,page,sentence_id,text) VALUES(?,?,?,?)", new_rows
                    )
                    seen_doc_ids.update(row[0] for row in new_rows)
                    inserted += len(new_rows)

            demo_db.execute("INSERT INTO evidence_fts(evidence_fts) VALUES('optimize')")
            demo_db.commit()
        finally:
            demo_db.close()
        temporary.replace(output)
    finally:
        source_db.close()

    metadata = {
        "sentences": inserted,
        "terms": None,
        "index_bytes": output.stat().st_size,
        "source": "FEVER Wikipedia June 2017 snapshot",
        "source_sentence_count": total,
        "limit": inserted,
        "deployment_demo": True,
        "random_sample_sentences": sample_size,
        "seeded_query_sentences": inserted - sample_size,
        "sampling": "Deterministic random sentence sample plus top BM25 matches for demo topics",
        "seed": seed,
        "demo_topics": list(DEMO_QUERIES),
        "backend": "SQLite FTS5 / BM25",
    }
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Path to the full FEVER SQLite FTS5 index")
    parser.add_argument("--output", type=Path, default=Path("data/demo/fever_demo.sqlite3"))
    parser.add_argument("--size", type=int, default=100_000, help="Number of sentence records in the demo index")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for reproducible row sampling")
    args = parser.parse_args()
    if args.size < 1:
        parser.error("--size must be greater than zero")
    print(json.dumps(build_demo_index(args.source, args.output, args.size, args.seed), indent=2), flush=True)


if __name__ == "__main__":
    main()
