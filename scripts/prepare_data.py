"""Inspect official FEVER JSONL files; does not download large corpora."""
import argparse, json
from pathlib import Path
from src.data.load_data import load_claims
from src.data.validate_data import summarize_claims
from src.utils.logging_config import configure_logging

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("claims", help="FEVER claims JSONL")
    parser.add_argument("--output", default="artifacts/results/dataset_summary.json")
    args = parser.parse_args(); configure_logging()
    summary = summarize_claims(load_claims(args.claims))
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2))

if __name__ == "__main__": main()
