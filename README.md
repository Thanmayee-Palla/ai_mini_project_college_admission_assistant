# College Admission Assistant

An AI-powered RAG chatbot that answers any question about college admissions in India — eligibility, documents, deadlines, fees, cutoff ranks, step-by-step checklists, and more.

Powered by: LangGraph · ChromaDB · SentenceTransformers · Groq (LLaMA 3.1) · FastAPI

---

## Features

- **Rank-based admission check** — Type your JEE / NEET / CUET rank and get an instant verdict
- **Eligibility guidance** — Entrance exam requirements, minimum 12th percentage
- **Fee structure** — UG, PG fee ranges and scholarship info per stream
- **Document checklist** — Stream-wise (Engineering / Medical / Commerce / Arts)
- **Step-by-step admission process** — With key deadlines
- **Placement & ratings** — Academics, faculty, infrastructure scores
- **Contact details** — Phone, email per college
- **12 colleges covered** — IIT Delhi, IIT Bombay, VIT Vellore, SRM Institute, Anna University, BITS Pilani, NIT Trichy, Amity University, Delhi University, Jadavpur University, Christ University, Mumbai University
- **350-row dataset** — Real admission records with rank ranges, fees, deadlines

---

## Project Structure

```
college_admission_assistant/
├── backend/
│   ├── main.py                  # FastAPI server
│   ├── graph.py                 # LangGraph pipeline (7 nodes)
│   ├── ingest.py                # CSV -> ChromaDB ingestion
│   ├── requirements.txt
│   ├── .env                     # GROQ_API_KEY goes here
│   ├── data/
│   │   └── college_dataset_350.csv
│   └── utils/
│       ├── intent_parser.py     # Keyword intent detection
│       ├── college_matcher.py   # RapidFuzz college name matching
│       └── synthesizer.py       # Rule-based context augmentation
└── frontend/
    └── index.html               # Single-file chat UI
```

---

## Setup

### 1. Install dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set your Groq API key

Edit `backend/.env`:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at https://console.groq.com

### 3. Build the vector database (run once)

```bash
cd backend
python ingest.py
```

This reads `data/college_dataset_350.csv` and builds a ChromaDB vector store in `backend/chroma_db/`.

### 4. Start the backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 5. Open the frontend

Open `frontend/index.html` in any browser (no server needed — it calls the API at localhost:8000).

---

## Example Questions

- "My JEE rank is 8500 — can I get CSE at NIT Trichy?"
- "What documents do I need for VIT Vellore engineering admission?"
- "What are the fees at IIT Delhi for BTech?"
- "What is the NEET cutoff for MBBS at SRM Institute?"
- "Give me a step-by-step admission checklist for engineering"
- "Which college should I choose with JEE rank 12000?"
- "What are the placements and ratings at Anna University?"
- "When is the application deadline for BITS Pilani?"

---

## LangGraph Pipeline

```
parse_intent → rank_resolver → [clarification | rag_retriever]
                                         ↓
                                  confidence_check
                                         ↓
                                  synthesize_flow
                                         ↓
                                  generate_response
```

- **parse_intent** — Detects intent (fees / cutoff / eligibility / admission / documents / placements / contact / courses)
- **rank_resolver** — Extracts numeric rank and exam type from query
- **rag_retriever** — Semantic search in ChromaDB with intent + college filters
- **confidence_check** — Retries with broader filter if confidence is low
- **synthesize_flow** — Rule-based keyword extraction to supplement sparse RAG chunks
- **generate_response** — LLaMA 3.1 via Groq with rank-aware prompting

---

## API

### POST /chat
```json
{ "session_id": "optional-uuid", "message": "your question" }
```
Response:
```json
{
  "session_id": "uuid",
  "answer": "...",
  "intent": "cutoff",
  "college_filter": "NIT Trichy",
  "user_rank": 8500
}
```

### DELETE /session/{id} — clear conversation history
### GET /health — service health check
