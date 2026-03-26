import sys
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Đường nối các mảnh ghép
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.generator import generate_answer
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. XÂY TRẠM PHÁT SÓNG FASTAPI ---
app = FastAPI(
    title="Luật Sư Ảo RAG API",
    description="Cổng giao tiếp Internet siêu tốc cho RAG Chatbot",
    version="2.0.0"
)

# Cho phép Frontend gọi API mà không bị trình duyệt chặn (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # Cho phép mọi nguồn gốc (dev mode)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Khởi động trước bộ nhớ DB để ghim cứng lên RAM
from app.services.vector_db import get_vector_db
print(">>> FASTAPI ĐANG BẬT MÁY: Ráp nối mô hình AI vào RAM trước...")
get_vector_db()
print(">>> MÁY SẴN SÀNG!")

# --- 2. BẢN THIẾT KẾ ĐẦU VÀO ĐẦU RA ---
class ChatRequest(BaseModel):
    query: str
    chat_history: List[dict] = []

class ChatResponse(BaseModel):
    answer: str

# --- 3. ĐƯỜNG HẦM API ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Nhận câu hỏi từ Frontend, xử lý RAG, trả kết quả."""
    lc_history = []
    for tin_nhan in request.chat_history:
        if tin_nhan.get("role") == "user":
            lc_history.append(HumanMessage(content=tin_nhan.get("content")))
        elif tin_nhan.get("role") == "assistant":
            lc_history.append(AIMessage(content=tin_nhan.get("content")))
            
    cau_tra_loi = generate_answer(query=request.query, chat_history=lc_history)
    return ChatResponse(answer=cau_tra_loi)

# --- 4. PHỤC VỤ GIAO DIỆN WEB (Frontend) ---
# Trỏ đường dẫn tĩnh vào thư mục ui
ui_dir = os.path.join(os.path.dirname(__file__), "ui")

@app.get("/")
def serve_frontend():
    """Mở trang chủ sẽ hiện ra giao diện Chat siêu xịn."""
    return FileResponse(os.path.join(ui_dir, "index.html"))

@app.get("/health")
def health_check():
    return {"status": "alive", "message": "Trạm phát sóng API đang hoạt động bình thường!"}
