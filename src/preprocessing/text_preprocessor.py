"""Text cleaning, tokenization, stop-word filtering and optional stemming."""
from __future__ import annotations
import re
import unicodedata
from functools import lru_cache

TOKEN_RE = re.compile(r"[\w]+(?:['’][\w]+)?", re.UNICODE)
DEFAULT_CONFIG = {"remove_stopwords": False, "stemming": False}

@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    # sklearn is already a core dependency. Preserve negation because it can
    # reverse the meaning of a fact-verification claim.
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    return set(ENGLISH_STOP_WORDS) - {"no", "not", "nor", "never"}

@lru_cache(maxsize=1)
def _porter_stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()

def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text).casefold()

def tokenize(text: str, *, remove_stopwords: bool = False, stemming: bool = False) -> list[str]:
    tokens = TOKEN_RE.findall(normalize(text))
    if remove_stopwords:
        tokens = [token for token in tokens if token not in _stopwords()]
    if stemming:
        stemmer = _porter_stemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def tokenize_with_config(text: str, config: dict | None = None) -> list[str]:
    settings = {**DEFAULT_CONFIG, **(config or {})}
    return tokenize(text, **settings)
