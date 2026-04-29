"""
main.py — FastAPI server for College Admission Assistant
Run: uvicorn main:app --reload --port 8000
"""

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from graph import GRAPH

app = FastAPI(
    title="College Admission Assistant API",
    description="RAG chatbot for Indian college admissions — eligibility, documents, fees, cutoffs and more.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
sessions: dict[str, dict] = {}


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    intent: str
    college_filter: str | None
    user_rank: int | None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {"message_history": [], "pending_query": None}

    session = sessions[session_id]

    initial_state = {
        "query": req.message.strip(),
        "intent": "general",
        "college_filter": None,
        "college_confidence": 0.0,
        "needs_clarification": False,
        "user_rank": None,
        "user_exam": None,
        "retrieved_chunks": [],
        "retrieval_confident": False,
        "synthesized_context": "",
        "response": "",
        "message_history": session["message_history"],
        "pending_query": session["pending_query"],
    }

    result = GRAPH.invoke(initial_state)

    session["message_history"].append({"role": "user",      "content": req.message})
    session["message_history"].append({"role": "assistant", "content": result["response"]})
    session["pending_query"] = result.get("pending_query")
    session["message_history"] = session["message_history"][-20:]

    return ChatResponse(
        session_id=session_id,
        answer=result["response"],
        intent=result["intent"],
        college_filter=result.get("college_filter"),
        user_rank=result.get("user_rank"),
    )


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"message": "Session cleared."}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "College Admission Assistant"}
