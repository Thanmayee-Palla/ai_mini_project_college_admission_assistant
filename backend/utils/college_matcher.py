"""
utils/college_matcher.py
Two-layer college name matching using RapidFuzz token_set_ratio + token overlap.
Returns (canonical_name, confidence, needs_clarification).
"""

import re
from rapidfuzz import fuzz

STOPWORDS = {
    "college", "of", "engineering", "and", "technology", "sciences",
    "science", "institute", "university", "for", "studies", "the",
    "in", "at",
}

FUZZY_WEIGHT            = 0.60
OVERLAP_WEIGHT          = 0.40
HIGH_CONFIDENCE_THRESHOLD = 55   
MARGIN_THRESHOLD          = 8    


def _tokenise(text: str) -> set:
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def match_college(query: str, known_colleges: list) -> tuple:
    """
    Returns (canonical_name, confidence_score, needs_clarification).
    needs_clarification=True means no college was clearly identified.
    """
    if not known_colleges:
        return None, 0.0, True

    query_tokens = _tokenise(query)
    scores = []

    query_lower = query.lower()
    for name in known_colleges:
        name_lower = name.lower()
        if name_lower in query_lower:
            return name, 100.0, False

    for name in known_colleges:
        fuzzy_score = fuzz.token_set_ratio(query.lower(), name.lower())
        name_tokens = _tokenise(name)
        if query_tokens and name_tokens:
            overlap = len(query_tokens & name_tokens) / max(len(name_tokens), 1)
        else:
            overlap = 0.0
        combined = FUZZY_WEIGHT * fuzzy_score + OVERLAP_WEIGHT * (overlap * 100)
        scores.append((name, combined))

    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score  = scores[0]
    second_score           = scores[1][1] if len(scores) > 1 else 0

    if best_score < HIGH_CONFIDENCE_THRESHOLD:
        return None, best_score, True

    if (best_score - second_score) < MARGIN_THRESHOLD:
        return None, best_score, True

    return best_name, best_score, False


def get_all_college_names(collection) -> list:
    """Extract unique canonical college names from ChromaDB collection."""
    results = collection.get(include=["metadatas"])
    names = {m["college_name"] for m in results["metadatas"]}
    return list(names)
