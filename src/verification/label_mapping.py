"""Map NLI labels to FEVER labels conservatively."""
def map_nli(entailment: float, contradiction: float, *, threshold: float = .6) -> str:
    if entailment >= threshold and entailment > contradiction: return "SUPPORTS"
    if contradiction >= threshold and contradiction > entailment: return "REFUTES"
    return "NOT ENOUGH INFO"
