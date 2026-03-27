# ⚖️ Trợ Lý Pháp Lý AI (RAG Chatbot)

Chatbot hỏi đáp về **Bộ Luật Lao Động Việt Nam 2019** theo kiến trúc **RAG (Retrieval-Augmented Generation)**.
Hệ thống kết hợp:
- Hybrid Retrieval (`Vector Search + BM25`)
- Cross-Encoder Reranking
- Router cho hội thoại nhiều lượt
- FastAPI backend + Web UI

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-5A67D8)

## 1) Tính năng chính

- Truy xuất lai (`Hybrid Search`): semantic search + keyword search.
- Rerank bằng `cross-encoder/ms-marco-MiniLM-L-6-v2` để tăng độ chính xác top-k.
- Router hội thoại: tự nhận biết khi nào cần dùng ngữ cảnh chat history.
- Sinh câu trả lời có ràng buộc theo tài liệu luật đã nạp.
- Lưu lịch sử phiên chat qua file `data/chat_history.json`.
- Có cả web UI (FastAPI serve HTML) và UI đơn giản qua Streamlit.

## 2) Kiến trúc tổng quan

```text
User
  -> FastAPI (/api/chat)
     -> Router (YES/NO: có cần ngữ cảnh hội thoại không)
        -> (nếu cần) Rewrite câu hỏi theo lịch sử chat
     -> Hybrid Retriever
        -> Vector Search (Chroma)
        -> BM25
        -> RRF Fusion
        -> Cross-Encoder Rerank
     -> LLM (Groq - Llama 3.x)
  <- Final answer
```

## 3) Cấu trúc thư mục

```text
RAG chatbot/
├─ app/
│  ├─ main.py
│  ├─ core/
│  │  └─ prompts.py
│  ├─ services/
│  │  ├─ document_loader.py
│  │  ├─ text_splitter.py
│  │  ├─ vector_db.py
│  │  ├─ hybrid_retriever.py
│  │  └─ generator.py
│  └─ ui/
│     ├─ index.html
│     ├─ app.py
│     └─ chat_ui.py
├─ data/
│  ├─ raw/
│  ├─ processed/
│  ├─ vector_store/
│  └─ chat_history.json
├─ tests/
│  ├─ test_rag.py
│  ├─ evaluate_rag.py
│  └─ evaluate_retrieval.py
├─ requirements.txt
└─ README.md
```

## 4) Yêu cầu môi trường

- Python `3.11+`
- `pip`
- Groq API key

## 5) Cài đặt nhanh

```bash
# 1) Tạo môi trường ảo
python -m venv venv

# 2) Kích hoạt (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# 3) Cài dependencies
pip install -r requirements.txt
```

Tạo file `.env` ở thư mục gốc:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 6) Chuẩn bị dữ liệu và tạo Vector DB

Đặt tài liệu luật vào `data/raw/` (ví dụ: `Bo_luat_lao_dong_2019.docx`), sau đó chạy:

```bash
python app/services/document_loader.py
python app/services/text_splitter.py
python app/services/vector_db.py
```

Ghi chú:
- `vector_db.py` có thể rebuild lại toàn bộ `data/vector_store`.
- Lần chạy đầu sẽ tải embedding model và cross-encoder nên có thể chậm hơn.

## 7) Chạy ứng dụng

### Cách A: FastAPI + Web UI chính

```bash
uvicorn app.main:app --reload
```

- Chat UI: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`

### Cách B: Streamlit UI (nhẹ)

```bash
streamlit run app/ui/app.py
```

## 8) API chính

### `POST /api/chat`

Request mẫu:

```json
{
  "query": "Lao động nữ nghỉ thai sản bao nhiêu tháng?",
  "session_id": "optional-session-id",
  "chat_history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Response mẫu:

```json
{
  "answer": "...",
  "session_id": "a1b2c3d4"
}
```

Các endpoint khác:
- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`

## 9) Đánh giá và kiểm thử

```bash
# Test pipeline hội thoại/router cơ bản
python tests/test_rag.py

# Đánh giá retrieval (precision/recall/mrr)
python tests/evaluate_retrieval.py

# Đánh giá end-to-end bằng LLM-as-a-judge
python tests/evaluate_rag.py
```

## 10) Công nghệ sử dụng

- Backend: `FastAPI`, `Uvicorn`
- LLM: `langchain-groq` (Llama 3.x)
- Orchestration: `LangChain`
- Vector DB: `ChromaDB`
- Embedding: `FastEmbedEmbeddings` (model multilingual)
- Keyword retrieval: `rank-bm25`
- Reranker: `sentence-transformers` CrossEncoder
- UI: HTML/CSS/JS + Streamlit

## 11) Phạm vi dự án

Dự án hiện tập trung vào **Bộ Luật Lao Động Việt Nam 2019**.
Nếu câu hỏi ngoài phạm vi tài liệu đã nạp, bot sẽ từ chối hoặc trả lời giới hạn theo prompt an toàn.

## 12) License

Phục vụ mục đích học tập, nghiên cứu và demo kỹ thuật.
