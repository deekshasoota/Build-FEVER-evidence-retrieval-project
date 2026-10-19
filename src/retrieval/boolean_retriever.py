"""Simple Boolean retrieval supporting terms joined by AND, OR, and NOT."""
import re
from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessor import tokenize

def search(query: str, index: InvertedIndex) -> list[str]:
    parts = re.split(r"\s+(AND|OR|NOT)\s+", query, flags=re.I)
    if not parts or not parts[0].strip(): return []
    result = index.candidates(tokenize(parts[0]))
    i = 1
    while i < len(parts):
        op, terms = parts[i].upper(), index.candidates(tokenize(parts[i + 1]))
        if op == "AND": result &= terms
        elif op == "OR": result |= terms
        elif op == "NOT": result -= terms
        i += 2
    return sorted(result)
