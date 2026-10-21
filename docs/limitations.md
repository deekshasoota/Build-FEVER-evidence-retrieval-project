# Limitations

- No FEVER or Wikipedia data is bundled; access and licensing conditions are controlled by the official project.
- No full-data download or benchmark was performed.
- Interactive corpus-wide TF-IDF and sentence-embedding resources are capped at 100,000 sentences to manage memory. Full-scale ranking needs a disk-backed sparse/dense index.
- JSON persistence may be large and memory intensive at full scale; a compact SQLite or disk-backed postings format is a future improvement.
- Boolean parsing supports simple left-to-right operators, not parentheses or a complete query grammar.
- The system currently abstains on all verification claims; this is safe baseline behavior but not a completed fact verifier.
- Sentence embeddings and Hybrid RRF are implemented, but no labeled retrieval comparison has been generated in the checked-in project because the development claims file is not present. A complete 21-domain matrix also depends on the input carrying all 21 domain labels.
- Probability calibration, full evidence-set aggregation, official FEVER scorer integration, and rich error analysis remain future work.
