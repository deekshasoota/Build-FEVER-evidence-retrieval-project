# User guide

1. Install Python dependencies and obtain the official FEVER data from its project page.
2. Put claim and wiki JSONL files under `data/raw/`.
3. Inspect claims with `python -m scripts.prepare_data data/raw/train.jsonl`.
4. Build a manageable first index with `python -m scripts.build_index data/raw/wiki-pages.jsonl --limit 100000`.
5. Start the interface with `streamlit run app.py`.
6. Enter a claim and inspect the ranked sentences, scores, page titles, and sentence IDs.
7. To train the baseline, run `python -m scripts.train_model --train data/raw/train.jsonl --index artifacts/indexes/fever_index.json`; evaluate it only on development data with `python -m scripts.run_evaluation --dev data/raw/dev.jsonl --index artifacts/indexes/fever_index.json`.

The current prediction is an explicit abstention because the supervised verifier is not yet available. BM25 scores express ranking relevance, not confidence or proof. The sample fixture is only for checking the software path.
