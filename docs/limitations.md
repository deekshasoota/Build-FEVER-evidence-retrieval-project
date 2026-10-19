# Limitations

- No FEVER or Wikipedia data is bundled; access and licensing conditions are controlled by the official project.
- No full-data download or benchmark was performed.
- TF-IDF currently recomputes a vectorizer per query and is unsuitable for the full wiki corpus.
- JSON persistence may be large and memory intensive at full scale; a compact SQLite or disk-backed postings format is a future improvement.
- Boolean parsing supports simple left-to-right operators, not parentheses or a complete query grammar.
- The system currently abstains on all verification claims; this is safe baseline behavior but not a completed fact verifier.
- Semantic retrieval, calibration, full evidence-set aggregation, FEVER scorer integration, and rich error analysis remain future work.
