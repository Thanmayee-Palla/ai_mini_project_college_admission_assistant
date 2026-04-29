"""
utils/intent_parser.py
Keyword-based intent detection for the College Admission Assistant.
"""

INTENT_KEYWORDS: dict[str, list[str]] = {
    "documents": [
        "document", "documents", "certificate", "certificates", "marksheet",
        "marks memo", "tc", "transfer certificate", "id proof", "aadhaar", "aadhar",
        "photo", "photographs", "what do i need", "what should i bring",
        "required documents", "submission", "papers needed", "checklist",
    ],
    "fees": [
        "fee", "fees", "cost", "tuition", "hostel fee", "per year",
        "how much", "annual fee", "semester fee", "charge", "ug fee", "pg fee",
        "fee structure", "scholarship", "scholarships", "afford",
    ],
    "eligibility": [
        "eligible", "eligibility", "qualify", "qualification", "criteria",
        "percentage", "marks required", "minimum marks", "minimum percentage",
        "12th percentage", "score", "rank needed", "requirement",
        "what percentage", "what marks", "what score",
    ],
    "courses": [
        "course", "courses", "branch", "branches", "programme", "programs",
        "btech", "b.tech", "mtech", "m.tech", "mba", "mca", "bds", "mbbs",
        "bba", "bcom", "ba", "bjmc", "cse", "ece", "mechanical", "civil",
        "offered", "available", "specialization", "stream", "what courses",
    ],
    "admission": [
        "admission", "apply", "application", "procedure", "process",
        "how to join", "how to get", "steps", "counselling", "web options",
        "seat allotment", "register", "deadline", "last date", "application date",
        "when to apply", "application window", "how do i", "step by step",
    ],
    "placements": [
        "placement", "placements", "package", "salary", "lpa", "recruiter",
        "companies", "hiring", "job", "campus placement", "highest package",
        "rating", "ratings", "infrastructure", "faculty", "academics",
        "social life", "accommodation", "hostel", "review", "best college",
    ],
    "contact": [
        "contact", "phone", "email", "website", "address", "location",
        "reach", "call", "helpline", "number",
    ],
    "cutoff": [
        "cutoff", "cut off", "rank", "jee rank", "neet rank", "cuet rank", "cet rank",
        "last rank", "opening rank", "closing rank", "how much rank", "my rank",
        "will i get", "can i get", "rank cutoff", "what rank", "which college",
        "with rank", "for rank",
    ],
}

INTENT_FOCUS_INSTRUCTIONS: dict[str, str] = {
    "documents": "Focus on providing document submission requirements clearly.",
    "fees": "Focus on the fee structure, including UG/PG fees and scholarships if available.",
    "eligibility": "Focus on eligibility criteria, entrance exams required, and minimum percentages.",
    "courses": "Focus on the courses, branches, and specializations offered.",
    "admission": "Focus on describing the admission process, steps, and key deadlines.",
    "placements": "Focus on placement statistics, ratings, infrastructure, and faculty.",
    "contact": "Focus on providing contact information such as phone, email, and location.",
    "cutoff": (
        "Focus on cutoff and rank information per course and exam. "
        "If the user mentions a specific rank, compare it against the observed range "
        "and state clearly: 'Likely', 'Borderline', or 'Unlikely' for admission."
    ),
    "general": (
        "Provide a helpful, comprehensive, and concise answer based on the retrieved context. "
        "Address any questions the user asks about the colleges, courses, admissions, fees, or related dataset info."
    ),
}

# Topics that are clearly off-topic — return a polite refusal immediately
OFF_TOPIC_KEYWORDS = [
    "photosynthesis", "mitosis", "gravity", "newton", "einstein",
    "world war", "history", "recipe", "weather", "cricket score",
    "stock market", "poem", "joke", "movie", "song",
]


def detect_intent(query: str) -> str:
    query_lower = query.lower()

    # Off-topic check first
    if any(kw in query_lower for kw in OFF_TOPIC_KEYWORDS):
        return "off_topic"

    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in query_lower:
                return intent
    return "general"
