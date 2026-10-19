"""Create a reusable BM25 index from FEVER wiki-pages JSONL."""
import argparse
import json
import time
from pathlib import Path
from src.data.load_data import iter_wiki_sentences
from src.indexing.inverted_index import InvertedIndex
from src.utils.logging_config import configure_logging

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("wiki_jsonl"); p.add_argument("--limit", type=int)
    p.add_argument("--output", default="artifacts/indexes/fever_index.json")
    a = p.parse_args(); configure_logging(); started = time.perf_counter()
    index = InvertedIndex()
    count = 0
    for row in iter_wiki_sentences(a.wiki_jsonl, limit=a.limit):
        index.add(row); count += 1
        if count % 100000 == 0: print(f"Indexed {count} sentences…")
    index.save(a.output)
    elapsed = time.perf_counter() - started
    output = Path(a.output)
    metadata = {"sentences": count, "terms": len(index.postings), "build_seconds": elapsed,
                "index_bytes": output.stat().st_size, "source": str(Path(a.wiki_jsonl)), "limit": a.limit}
    output.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__": main()
