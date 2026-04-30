"""
graph.py — LangGraph agentic pipeline for College Admission Assistant

Node order:
  parse_intent -> rank_resolver -> clarification_check -> rag_retriever
               -> confidence_check -> synthesize_flow -> generate_response
"""

import os
import re
from typing import TypedDict, Optional

import chromadb
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from sentence_transformers import SentenceTransformer

from utils.college_matcher import match_college, get_all_college_names
from utils.intent_parser import detect_intent, INTENT_FOCUS_INSTRUCTIONS
from utils.synthesizer import synthesize_flow

load_dotenv()

#  Singletons
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
_chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection    = _chroma_client.get_collection("colleges")
_embed_model   = SentenceTransformer("all-MiniLM-L6-v2")
_llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2,
    max_tokens=1024,
)

KNOWN_COLLEGES = get_all_college_names(_collection)

HIGH_CONFIDENCE_DISTANCE = 0.40
MAX_HISTORY_TURNS = 4   

KNOWN_EXAMS = ["jee", "neet", "cuet", "cet"]

SYSTEM_PROMPT = """You are a College Admission Assistant for India. Answer the current question based ONLY on the retrieved context below.

STRICT RULES:
1. Answer ONLY from the retrieved context. Never invent fees, ranks, or documents.
2. If data is not in the context, say: "I don't have that specific information. Please check the official college website."
3. Be concise and structured. Use bullet points or numbered lists where appropriate.
4. When a user gives a rank, compare it to the observed cutoff ranges and give a clear verdict: "Likely", "Borderline", or "Unlikely". Remember that a LOWER numerical rank is BETTER (e.g. rank 1000 is better than rank 5000). If the user's rank is mathematically smaller than or within the cutoff limit, they are Likely to get in.
5. For admission process questions, give a clear step-by-step explanation if the context supports it.
6. Address any questions asked about the colleges in the dataset, including general queries combining multiple factors (e.g. fees and placements together).
7. If the user asks a question that requires a specific college (like fees) but doesn't mention one, use the retrieved context to inform them of some options and politely ask them to specify which college they mean.
"""


# State 
class ChatState(TypedDict):
    query: str
    intent: str
    college_filter: Optional[str]
    college_confidence: float
    needs_clarification: bool
    user_rank: Optional[int]
    user_exam: Optional[str]
    retrieved_chunks: list
    retrieval_confident: bool
    synthesized_context: str
    response: str
    message_history: list
    pending_query: Optional[str]


# Node 1: Parse Intent 
def parse_intent(state: ChatState) -> ChatState:
    query = state["query"]
    intent = detect_intent(query)
    college, confidence, needs_clarification = match_college(query, KNOWN_COLLEGES)

    # If answering a clarification, merge pending query
    if state.get("pending_query") and not needs_clarification:
        query = state["pending_query"] + " " + query
        state = {**state, "pending_query": None}

    return {
        **state,
        "query": query,
        "intent": intent,
        "college_filter": college,
        "college_confidence": confidence,
        "needs_clarification": needs_clarification,
    }


#  Node 2: Rank Resolver 
def rank_resolver(state: ChatState) -> ChatState:
    """Extract user rank and exam type from query if present."""
    query = state["query"]
    user_rank = None
    user_exam = None

    # Extract numeric rank 
    for match in re.finditer(r"\b(\d{1,6})\b", query):
        candidate = int(match.group(1))
        if 1 <= candidate <= 200000:
            user_rank = candidate
            break

    q_lower = query.lower()
    for exam in KNOWN_EXAMS:
        if exam in q_lower:
            user_exam = exam.upper()
            break

    
    needs_clarification = state["needs_clarification"]
    if user_rank and needs_clarification:
        needs_clarification = False  

    return {
        **state,
        "user_rank": user_rank,
        "user_exam": user_exam,
        "needs_clarification": needs_clarification,
    }


# Node 3: Clarification 
def clarification_node(state: ChatState) -> ChatState:
    college_list = ", ".join(sorted(KNOWN_COLLEGES)[:10])
    clarification_msg = (
        f"Could you please specify which college you're asking about?\n"
        f"I have information on: {college_list}, and more.\n\n"
        f"You can also ask things like:\n"
        f"• 'My JEE rank is 5000 — which college can I get?'\n"
        f"• 'What are the fees at VIT Vellore?'\n"
        f"• 'Give me the admission checklist for engineering'"
    )
    return {
        **state,
        "response": clarification_msg,
        "pending_query": state["query"],
    }


# Node 4: RAG Retriever 
def rag_retriever(state: ChatState) -> ChatState:
    query        = state["query"]
    intent       = state["intent"]
    college_filter = state["college_filter"]
    user_rank    = state.get("user_rank")
    user_exam    = state.get("user_exam")

    # Enrich embedding query with rank context
    embed_query = query
    if user_rank:
        embed_query += f" rank {user_rank}"
    if user_exam:
        embed_query += f" {user_exam}"

    query_vec = _embed_model.encode(embed_query).tolist()

    # Build ChromaDB where filter
    where = None
    if college_filter:
        where = {"college_name": {"$eq": college_filter}}

    try:
        results = _collection.query(
            query_embeddings=[query_vec],
            n_results=8,
            where=where,
            include=["documents", "distances"],
        )
        docs      = results["documents"][0]
        distances = results["distances"][0]
    except Exception:
        docs, distances = [], []

    best_distance = distances[0] if distances else 1.0
    retrieval_confident = best_distance < HIGH_CONFIDENCE_DISTANCE

    return {
        **state,
        "retrieved_chunks": docs,
        "retrieval_confident": retrieval_confident,
    }


#  Node 5: Confidence Check
def confidence_check(state: ChatState) -> ChatState:
    if state["retrieval_confident"] or not state["college_filter"]:
        return state

    query_vec = _embed_model.encode(state["query"]).tolist()
    try:
        results = _collection.query(
            query_embeddings=[query_vec],
            n_results=8,
            include=["documents", "distances"],
        )
        docs      = results["documents"][0]
        distances = results["distances"][0]
        best_distance = distances[0] if distances else 1.0
    except Exception:
        docs, distances, best_distance = [], [], 1.0

    return {
        **state,
        "retrieved_chunks": docs,
        "retrieval_confident": best_distance < HIGH_CONFIDENCE_DISTANCE,
    }


# Node 6: Synthesize Flow 
def synthesize_flow_node(state: ChatState) -> ChatState:
    synth = synthesize_flow(state["retrieved_chunks"], state["intent"])
    return {**state, "synthesized_context": synth}


# Node 7: Generate Response 
def generate_response(state: ChatState) -> ChatState:
    chunks    = state["retrieved_chunks"]
    synth     = state.get("synthesized_context", "")
    intent    = state["intent"]
    user_rank = state.get("user_rank")
    user_exam = state.get("user_exam")

    # Off-topic intent — refuse immediately, no LLM call needed
    if intent == "off_topic":
        return {**state, "response": "I can only help with college admission queries — fees, eligibility, cutoffs, documents, courses, and placements. Feel free to ask anything about college admissions!"}

    if not chunks:
        no_data_msg = (
            "I don't have specific information about that in my knowledge base. "
            "Please check the official college website for the most accurate details."
        )
        return {**state, "response": no_data_msg}

    context = "\n\n".join(chunks)
    if synth:
        context += synth

    focus = INTENT_FOCUS_INSTRUCTIONS.get(intent, INTENT_FOCUS_INSTRUCTIONS["general"])

    # Rank hint injected into system prompt, not history
    rank_hint = ""
    if user_rank:
        exam_str = f" in {user_exam}" if user_exam else ""
        rank_hint = (
            f"\n\nIMPORTANT: The user's rank{exam_str} is {user_rank:,}. "
            f"Compare this ONLY against the rank ranges in the retrieved context above "
            f"and state clearly: Likely / Borderline / Unlikely for each college/course found."
            f"\nCRITICAL RANK LOGIC: In entrance exams, a LOWER rank number is BETTER. "
            f"If the user's rank is LESS THAN or EQUAL TO the upper limit of the cutoff rank range (or the specific cutoff rank), they are LIKELY to get in. "
            f"If their rank is mathematically MUCH GREATER than the cutoff, they are UNLIKELY to get in."
        )

    system_msg = SYSTEM_PROMPT + f"\n\n{focus}" + rank_hint

  
    # For rank/cutoff queries, send NO history — answer must be self-contained
    if intent in ("cutoff", "eligibility") or user_rank:
        history_to_send = []
    else:
        history_to_send = state.get("message_history", [])[-4:]  # last 2 exchanges max

    messages = [SystemMessage(content=system_msg)]
    for msg in history_to_send:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    user_prompt = (
        f"Retrieved context:\n{context}\n\n"
        f"Current question (answer ONLY this): {state['query']}"
    )
    messages.append(HumanMessage(content=user_prompt))

    full_response = ""
    for attempt in range(3):
        result        = _llm.invoke(messages)
        part          = result.content.strip()
        full_response += part
        finish_reason = getattr(result, "response_metadata", {}).get("finish_reason", "stop")
        if finish_reason in ("length", "max_tokens") and attempt < 2:
            messages.append(AIMessage(content=part))
            messages.append(HumanMessage(content=(
                "Continue exactly from where you stopped. Do not repeat any previous lines."
            )))
        else:
            break

    return {**state, "response": full_response}


# Routing 
def route_after_rank_resolver(state: ChatState) -> str:
    return "rag_retriever"


def route_after_confidence(state: ChatState) -> str:
    return "synthesize_flow"


#  Build Graph 
def build_graph():
    builder = StateGraph(ChatState)

    builder.add_node("parse_intent",      parse_intent)
    builder.add_node("rank_resolver",     rank_resolver)
    builder.add_node("clarification",     clarification_node)
    builder.add_node("rag_retriever",     rag_retriever)
    builder.add_node("confidence_check",  confidence_check)
    builder.add_node("synthesize_flow",   synthesize_flow_node)
    builder.add_node("generate_response", generate_response)

    builder.set_entry_point("parse_intent")
    builder.add_edge("parse_intent", "rank_resolver")
    builder.add_conditional_edges("rank_resolver", route_after_rank_resolver, {
        "clarification": "clarification",
        "rag_retriever": "rag_retriever",
    })
    builder.add_edge("clarification",     END)
    builder.add_edge("rag_retriever",     "confidence_check")
    builder.add_conditional_edges("confidence_check", route_after_confidence, {
        "synthesize_flow": "synthesize_flow",
    })
    builder.add_edge("synthesize_flow",   "generate_response")
    builder.add_edge("generate_response", END)

    return builder.compile()


GRAPH = build_graph()