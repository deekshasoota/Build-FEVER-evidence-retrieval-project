"""Conservative normalization and tokenization; retain negation and numbers."""
from __future__ import annotations
import re
import unicodedata

TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text).casefold()

def tokenize(text: str, *, remove_stopwords: bool = False) -> list[str]:
    tokens = TOKEN_RE.findall(normalize(text))
    if remove_stopwords:
        # Deliberately keep not/no/never: negation is critical in verification.
        stop = {"a", "an", "the", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or"}
        tokens = [t for t in tokens if t not in stop]
    return tokens
