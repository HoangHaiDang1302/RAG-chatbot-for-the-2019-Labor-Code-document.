# ⚖️ Trợ Lý Pháp Lý AI (RAG Chatbot)

Hệ thống chatbot hỏi đáp về **Bộ Luật Lao Động Việt Nam 2019** theo kiến trúc **RAG (Retrieval-Augmented Generation)**, tối ưu cho:
- Độ chính xác cao trên dữ liệu pháp lý.
- Độ trễ thấp với tập dữ liệu nhỏ (khoảng >200 chunks).
- Dễ vận hành trên máy local và dễ mở rộng sau này.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-5A67D8)

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Tính năng nổi bật](#2-tính-năng-nổi-bật)
3. [Kiến trúc hệ thống](#3-kiến-trúc-hệ-thống)
4. [Cấu trúc thư mục](#4-cấu-trúc-thư-mục)
5. [Yêu cầu môi trường](#5-yêu-cầu-môi-trường)
6. [Cài đặt nhanh](#6-cài-đặt-nhanh)
7. [Chuẩn bị dữ liệu và build Vector DB](#7-chuẩn-bị-dữ-liệu-và-build-vector-db)
8. [Chạy ứng dụng](#8-chạy-ứng-dụng)
9. [API chi tiết](#9-api-chi-tiết)
10. [Chiến lược cache và tối ưu latency](#10-chiến-lược-cache-và-tối-ưu-latency)
11. [Đánh giá chất lượng](#11-đánh-giá-chất-lượng)
12. [Vận hành và quan sát](#12-vận-hành-và-quan-sát)
13. [Troubleshooting](#13-troubleshooting)
14. [Roadmap đề xuất](#14-roadmap-đề-xuất)
15. [Phạm vi dự án](#15-phạm-vi-dự-án)
16. [License](#16-license)

## 1) Tổng quan

Dự án sử dụng pipeline RAG để trả lời câu hỏi pháp lý dựa trên nguồn văn bản đã nạp.  
Mục tiêu chính:
- Không trả lời theo kiểu suy diễn ngoài tài liệu.
- Ưu tiên tốc độ phản hồi tốt cho nhu cầu demo/thực chiến nhỏ.
- Cho phép theo dõi hiệu quả cache theo thời gian thực.

Các thành phần cốt lõi:
- `FastAPI` cho backend API + web entrypoint.
- `ChromaDB` cho vector storage.
- `BM25` để tăng khả năng khớp từ khóa pháp lý.
- `Cross-Encoder` để rerank top candidate.
- `Groq` (Llama 3.x) cho bước tạo câu trả lời.

## 2) Tính năng nổi bật

- **Hybrid Retrieval**: Vector Search + BM25.
- **Reranking**: dùng `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Router hội thoại**:
  - Rule-based router (nhanh).
  - Fallback LLM khi câu hỏi ở vùng mơ hồ.
- **3 lớp cache runtime**:
  - Context cache: cache quyết định router + câu hỏi rewrite.
  - Retrieval cache: cache kết quả `hybrid_search`.
  - Answer cache: cache câu trả lời hoàn chỉnh.
- **Session management**: lưu lịch sử hội thoại vào `data/chat_history.json`.
- **Metrics API**: endpoint theo dõi hit/miss của cache.
- **Hai UI**:
  - FastAPI phục vụ `index.html`.
  - Streamlit UI đơn giản.

## 3) Kiến trúc hệ thống

```text
User
  -> FastAPI (/api/chat)
     -> Answer Cache
       -> Context Cache
         -> Router (Rule-based)
            -> Fallback LLM (nếu mơ hồ)
            -> Rewrite query (nếu cần)
         -> Retrieval Cache
            -> Hybrid Retriever
               -> Vector Search (Chroma)
               -> BM25 Search
               -> RRF Fusion
               -> Cross-Encoder Rerank
         -> LLM Generation (Groq)
  <- Final answer
```

Luồng xử lý chuẩn:
1. Nhận `query`, `session_id`, `chat_history`.
2. Kiểm tra `answer cache`.
3. Nếu miss:
   - Kiểm tra `context cache`.
   - Router quyết định có cần ngữ cảnh hội thoại hay không.
   - Rewrite query khi cần.
4. Gọi retrieval (có retrieval cache).
5. Gọi LLM tạo câu trả lời cuối.
6. Lưu kết quả vào cache và lịch sử hội thoại.

## 4) Cấu trúc thư mục

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

## 5) Yêu cầu môi trường

- Python `3.11+`
- `pip`
- Groq API key

## 6) Cài đặt nhanh

```bash
# 1) Tạo virtual environment
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

## 7) Chuẩn bị dữ liệu và build Vector DB

### Bước 1: Đặt tài liệu nguồn

Đặt file luật vào `data/raw/`, ví dụ:
- `Bo_luat_lao_dong_2019.docx`

### Bước 2: Chạy pipeline xử lý

```bash
python app/services/document_loader.py
python app/services/text_splitter.py
python app/services/vector_db.py
```

Ghi chú:
- Bước đầu có thể chậm do tải model embedding/reranker.
- `vector_db.py` có thể rebuild lại hoàn toàn `data/vector_store`.

## 8) Chạy ứng dụng

### Cách A: FastAPI + Web UI

```bash
uvicorn app.main:app --reload
```

- Chat UI: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`

### Cách B: Streamlit UI

```bash
streamlit run app/ui/app.py
```

## 9) API chi tiết

### 9.1 `POST /api/chat`

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

### 9.2 Session APIs

- `GET /api/sessions`
- `GET /api/sessions/{session_id}`
- `DELETE /api/sessions/{session_id}`

### 9.3 Metrics API

- `GET /api/metrics/cache`

Response ví dụ:

```json
{
  "generator": {
    "answer_cache_hit": 10,
    "answer_cache_miss": 20,
    "context_cache_hit": 8,
    "context_cache_miss": 12,
    "router_rule_yes": 6,
    "router_rule_no": 4,
    "router_llm_fallback": 2
  },
  "retrieval": {
    "retrieval_cache_hit": 9,
    "retrieval_cache_miss": 11
  }
}
```

## 10) Chiến lược cache và tối ưu latency

### 10.1 Cấu hình `.env`

```env
# Context cache
CONTEXT_CACHE_TTL_SECONDS=900
CONTEXT_CACHE_MAX_ITEMS=512

# Retrieval cache
RETRIEVAL_CACHE_TTL_SECONDS=900
RETRIEVAL_CACHE_MAX_ITEMS=1024

# Answer cache
ANSWER_CACHE_TTL_SECONDS=600
ANSWER_CACHE_MAX_ITEMS=512

# Router thresholds
ROUTER_YES_THRESHOLD=0.65
ROUTER_NO_THRESHOLD=0.35

# 1 = chỉ dùng rule-based router, không fallback LLM
ROUTER_RULE_ONLY=0
```

### 10.2 Gợi ý profile cấu hình

**Ưu tiên tốc độ tối đa**

```env
ROUTER_RULE_ONLY=1
ANSWER_CACHE_TTL_SECONDS=1800
RETRIEVAL_CACHE_TTL_SECONDS=1800
CONTEXT_CACHE_TTL_SECONDS=1800
```

**Cân bằng tốc độ/chất lượng**

```env
ROUTER_RULE_ONLY=0
ROUTER_YES_THRESHOLD=0.65
ROUTER_NO_THRESHOLD=0.35
ANSWER_CACHE_TTL_SECONDS=600
RETRIEVAL_CACHE_TTL_SECONDS=900
```

### 10.3 Đọc metrics để tune

- `answer_cache_hit` tăng: user hỏi lặp nhiều, độ trễ giảm rõ.
- `retrieval_cache_hit` tăng: giảm chi phí vector/BM25/rerank.
- `router_llm_fallback` cao: rule-based còn mơ hồ nhiều, cần chỉnh ngưỡng hoặc rule.

## 11) Đánh giá chất lượng

```bash
# Kiểm thử pipeline cơ bản
python tests/test_rag.py

# Benchmark retrieval (precision/recall/mrr)
python tests/evaluate_retrieval.py

# Đánh giá end-to-end bằng LLM-as-a-judge
python tests/evaluate_rag.py
```

Khuyến nghị:
- Chạy benchmark retrieval trước và sau mỗi thay đổi lớn ở retriever.
- Theo dõi đồng thời chất lượng và latency, tránh chỉ tối ưu một phía.

## 12) Vận hành và quan sát

Checklist nhanh khi chạy demo:
1. Kiểm tra `GROQ_API_KEY`.
2. Warm-up 1-2 query sau khi khởi động server.
3. Gọi `GET /api/metrics/cache` sau vài phiên để xem hit/miss.
4. Nếu `router_llm_fallback` cao, cân nhắc tăng tính quyết đoán của rule-based.

## 13) Troubleshooting

### Lỗi `Chưa có chìa khóa GROQ_API_KEY`

- Kiểm tra file `.env`.
- Khởi động lại server sau khi thay đổi biến môi trường.

### Không có dữ liệu để truy xuất

- Kiểm tra tài liệu có trong `data/raw/`.
- Chạy lại pipeline mục 7.

### Request đầu tiên chậm

- Do warm-up model/index, đây là bình thường.

### Kết quả truy xuất chưa tốt

- Kiểm tra chiến lược chunking.
- Tune `top_k` và rerank policy trong `hybrid_retriever.py`.
- So sánh bằng `tests/evaluate_retrieval.py`.

## 14) Roadmap đề xuất

1. Thêm latency breakdown theo từng stage (router/retrieval/generation).
2. Persist cache sang Redis để giữ qua restart.
3. Chuẩn hóa metadata pháp lý (Điều/Khoản/Chương) để filter retrieval tốt hơn.
4. Bổ sung bộ test case thực tế theo nghiệp vụ pháp lý Việt Nam.

## 15) Phạm vi dự án

Dự án hiện tập trung vào **Bộ Luật Lao Động Việt Nam 2019**.
Nếu câu hỏi ngoài phạm vi tài liệu đã nạp, hệ thống trả lời giới hạn theo prompt an toàn.

## 16) License

Phục vụ học tập, nghiên cứu và demo kỹ thuật.
