import json
import os
import sys
import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.generator import generate_answer, get_runtime_metrics
from app.services.hybrid_retriever import get_retrieval_runtime_metrics
from app.services.vector_db import get_vector_db
from langchain_core.messages import AIMessage, HumanMessage

app = FastAPI(title="Luật Sư Ảo RAG API", version="3.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(">>> FASTAPI ĐANG BẬT MÁY...")
get_vector_db()
print(">>> MÁY SẴN SÀNG!")

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "chat_history.json")


def _load_all_sessions():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_all_sessions(sessions):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(sessions, f, ensure_ascii=False, indent=2)


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    chat_history: List[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


class SessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: str
    message_count: int


@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())[:8]

    lc_history = []
    for msg in request.chat_history:
        if msg.get("role") == "user":
            lc_history.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            lc_history.append(AIMessage(content=msg.get("content")))

    answer = generate_answer(query=request.query, chat_history=lc_history)

    sessions = _load_all_sessions()
    if session_id not in sessions:
        sessions[session_id] = {
            "title": request.query[:50],
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "messages": [],
        }

    sessions[session_id]["messages"].append({"role": "user", "content": request.query})
    sessions[session_id]["messages"].append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
    _save_all_sessions(sessions)

    return ChatResponse(answer=answer, session_id=session_id)


@app.get("/api/sessions")
def get_all_sessions():
    sessions = _load_all_sessions()
    result = []
    for sid, data in sessions.items():
        result.append(
            SessionSummary(
                session_id=sid,
                title=data.get("title", "Cuộc trò chuyện"),
                created_at=data.get("created_at", ""),
                message_count=len(data.get("messages", [])),
            )
        )
    result.sort(key=lambda x: x.created_at, reverse=True)
    return result


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    sessions = _load_all_sessions()
    if session_id in sessions:
        return sessions[session_id]
    return {"error": "Không tìm thấy cuộc hội thoại"}


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    sessions = _load_all_sessions()
    if session_id in sessions:
        del sessions[session_id]
        _save_all_sessions(sessions)
        return {"message": "Đã xóa"}
    return {"error": "Không tìm thấy"}


@app.get("/api/metrics/cache")
def get_cache_metrics():
    return {
        "generator": get_runtime_metrics(),
        "retrieval": get_retrieval_runtime_metrics(),
    }


ui_dir = os.path.join(os.path.dirname(__file__), "ui")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(ui_dir, "index.html"))
