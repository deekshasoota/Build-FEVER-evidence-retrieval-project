"""Train verifier from FEVER training labels and retrieved evidence only."""
import argparse
import logging
from src.verification.train_classifier import train

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True); p.add_argument("--index", required=True)
    p.add_argument("--limit", type=int); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="artifacts/models/verifier.pkl")
    a = p.parse_args(); logging.basicConfig(level=logging.INFO)
    print(train(a.train, a.index, a.output, limit=a.limit, seed=a.seed))

if __name__ == "__main__": main()
