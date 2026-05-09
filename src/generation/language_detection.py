# src/generation/language_detection.py
from __future__ import annotations

import re

# Stopwords that are unambiguously French — chosen to avoid overlap with
# common English tokens (e.g. "a" is excluded because it appears in both).
_FR_STOPWORDS: frozenset[str] = frozenset(
    {
        # articles
        "le",
        "la",
        "les",
        "un",
        "une",
        "des",
        # prepositions / contractions
        "de",
        "du",
        "au",
        "aux",
        "par",
        "pour",
        "sur",
        "sous",
        "dans",
        "avec",
        "sans",
        # subject pronouns
        "je",
        "tu",
        "il",
        "elle",
        "nous",
        "vous",
        "ils",
        "elles",
        # determiners / possessives
        "ce",
        "cet",
        "cette",
        "ces",
        "mon",
        "ma",
        "mes",
        "ton",
        "ta",
        "tes",
        "son",
        "sa",
        "ses",
        # conjunctions / relative pronouns
        "et",
        "que",
        "qui",
        "où",
        # common verb form
        "est",
        # question words
        "comment",
        "pourquoi",
        "quel",
        "quelle",
        "quels",
        "quelles",
        # adverbs
        "très",
        "bien",
        "plus",
        "moins",
        "aussi",
    }
)

# Stopwords that are unambiguously English.
_EN_STOPWORDS: frozenset[str] = frozenset(
    {
        # articles
        "the",
        "a",
        "an",
        # prepositions
        "of",
        "in",
        "at",
        "by",
        "for",
        "with",
        "about",
        "from",
        "to",
        # subject pronouns
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "it",
        # object / possessive pronouns
        "me",
        "him",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "our",
        "their",
        "its",
        # conjunctions / determiners
        "and",
        "or",
        "but",
        "if",
        "that",
        "this",
        "these",
        "those",
        # auxiliary verbs
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        # question words
        "what",
        "which",
        "who",
        "when",
        "where",
        "how",
        "why",
        # negation
        "not",
        "no",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ]+")


def detect_language(query: str) -> str:
    """Detect whether *query* is French or English using stopword counting.

    Tokenises the query (lowercased, alphabetic tokens only), counts how many
    tokens match the French and English stopword sets, and returns the winning
    language code.  Defaults to ``"fr"`` on a tie or when no stopwords are
    found — consistent with the engine's primary deployment language.

    The detection is intentionally lightweight: it relies on high-frequency
    function words that appear in virtually every natural-language sentence
    and have no cross-language ambiguity.  This avoids any external
    dependency while being accurate enough for short RAG queries.

    Args:
        query: The user query string.  May be empty.

    Returns:
        ``"fr"`` if French stopwords outnumber English ones (or on a tie),
        ``"en"`` otherwise.

    Examples::

        >>> detect_language("Quelle est la capitale de la France ?")
        'fr'
        >>> detect_language("What is the capital of France?")
        'en'
        >>> detect_language("")
        'fr'
    """
    tokens = _TOKEN_RE.findall(query.lower())
    fr_count = sum(1 for t in tokens if t in _FR_STOPWORDS)
    en_count = sum(1 for t in tokens if t in _EN_STOPWORDS)
    return "en" if en_count > fr_count else "fr"
