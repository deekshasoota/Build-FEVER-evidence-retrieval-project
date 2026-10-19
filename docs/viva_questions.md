# Viva questions and answers

**What is an inverted index?** A mapping from each term to the evidence records containing it, so lookup avoids scanning every sentence.

**How do TF-IDF and cosine similarity work?** TF-IDF weights a term by within-document frequency and inverse corpus frequency; cosine compares the directions of claim and evidence vectors.

**Why BM25?** It is a strong lexical ranking function with term-frequency saturation and document-length normalization.

**What is the difference between precision and recall?** Precision measures how many retrieved items are relevant; recall measures how many relevant items were found.

**What are MRR and nDCG?** MRR rewards an early first relevant result. nDCG rewards relevant results near the top and supports graded relevance.

**Why is retrieval score not a verdict?** Similar wording can describe a contradiction or an unrelated fact; entailment needs claim-evidence reasoning.

**What does NOT ENOUGH INFO mean?** The supplied evidence does not establish support or refutation. It does not mean the claim is false.

**Why keep negation and numbers?** A token such as “not” or a changed date/quantity can reverse a claim's meaning.

**What is evidence-set completeness?** Whether retrieved evidence contains all sentences in at least one acceptable gold evidence group; it is stricter than sentence-level relevance.

**What does the FEVER score measure?** A claim counts as correct when its label is right and, for SUPPORTS or REFUTES, at least one complete acceptable evidence group is returned. A correct NOT ENOUGH INFO label does not require evidence.

**What is the current verifier's main limitation?** It abstains on all claims until a supervised evidence-pair model is implemented and evaluated on official data.
