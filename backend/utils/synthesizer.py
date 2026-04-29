"""
utils/synthesizer.py
Rule-based synthesiser: scans raw college chunks and extracts
intent-relevant lines to augment LLM context when RAG chunks are sparse.
"""

INTENT_KEYWORDS_SYNTHESIZE: dict[str, list[str]] = {
    "documents": [
        "certificate", "marksheet", "marks memo", "tc", "transfer",
        "id proof", "aadhaar", "aadhar", "photo", "passport", "document",
        "income", "caste", "study certificate", "migration",
    ],
    "fees": ["fee", "fees", "tuition", "hostel", "quota", "per year", "cost", "scholarship", "rs."],
    "eligibility": ["eligible", "eligibility", "percentage", "qualify", "exam", "jee", "neet", "cuet", "cet"],
    "admission": ["admission", "apply", "step", "counselling", "allotment", "register", "report", "deadline"],
    "courses": ["b.tech", "btech", "m.tech", "mba", "mca", "bds", "mbbs", "bba", "bcom", "course", "branch", "programme", "stream"],
    "placements": ["placement", "rating", "score", "academics", "faculty", "infrastructure", "social"],
    "cutoff": ["cutoff", "rank", "opening", "closing", "rank range", "admitted"],
    "contact": ["phone", "email", "website", "contact"],
    "general": [],
}


def synthesize_flow(college_chunks: list, intent: str) -> str:
    """
    Given raw text chunks for a college and detected intent,
    return a supplemental list of relevant lines.
    Returns empty string if nothing useful is found.
    """
    keywords = INTENT_KEYWORDS_SYNTHESIZE.get(intent, [])
    if not keywords:
        return ""

    matched_lines = []
    seen = set()
    for chunk in college_chunks:
        for line in chunk.split(". "):
            line = line.strip().rstrip(".")
            if not line:
                continue
            line_lower = line.lower()
            if any(kw in line_lower for kw in keywords):
                if line_lower not in seen:
                    seen.add(line_lower)
                    matched_lines.append(line)

    if not matched_lines:
        return ""

    numbered = "\n".join(f"{i+1}. {l}" for i, l in enumerate(matched_lines[:12]))
    return f"\n\n[Supplemental context for '{intent}']:\n{numbered}"
