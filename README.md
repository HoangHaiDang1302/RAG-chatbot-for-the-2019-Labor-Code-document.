# ⚖️ Trợ Lý Pháp Lý AI — RAG Chatbot

Ứng dụng Chatbot hỏi đáp thông minh về **Bộ Luật Lao Động Việt Nam 2019**, được xây dựng bằng kiến trúc **RAG (Retrieval-Augmented Generation)** kết hợp Hybrid Search, Reranking và Conversational Memory.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-latest-orange)
![Groq](https://img.shields.io/badge/Groq-Llama_3-purple)

---

## ✨ Tính năng nổi bật

| Tính năng | Mô tả |
|-----------|-------|
| 🔍 **Hybrid Search** | Kết hợp Vector Search (tìm theo ý nghĩa) + BM25 (tìm theo từ khóa chính xác) |
| 📊 **LLM Reranker** | Dùng AI chấm điểm lại các kết quả tìm kiếm, chọn ra top chính xác nhất |
| 🧠 **Conversational Memory** | Bot nhớ lịch sử hội thoại, hiểu được các câu hỏi ám chỉ ngữ cảnh trước đó |
| ⚡ **Smart Router** | Tự động phát hiện câu hỏi có cần đọc lại lịch sử hay không, tối ưu latency |
| 🚀 **FastAPI Backend** | API tốc độ cao, pre-load model vào RAM, hỗ trợ Swagger UI |
| 🎨 **Giao diện Dark Mode** | Frontend HTML/CSS/JS xịn xò với glassmorphism, animation mượt mà |
| 📝 **Auto Evaluation** | Bộ đánh giá chất lượng tự động (LLM-as-Judge) với 3 tiêu chí |

---

## 🏗️ Kiến trúc hệ thống

```
User (Browser)
     │
     ▼
┌─────────────────────────────────────────────────┐
│                  FastAPI Server                  │
│                  (app/main.py)                   │
│  ┌───────────────────────────────────────────┐  │
│  │             Generator Pipeline             │  │
│  │                                            │  │
│  │  ┌──────────┐    ┌─────────────────────┐  │  │
│  │  │ Smart    │    │  Hybrid Retriever   │  │  │
│  │  │ Router   │    │  ┌───────┬────────┐ │  │  │
│  │  │ (8B)     │    │  │Vector │  BM25  │ │  │  │
│  │  └────┬─────┘    │  │Search │ Search │ │  │  │
│  │       │          │  └───┬───┴───┬────┘ │  │  │
│  │  ┌────▼─────┐    │      │  Reranker  │ │  │  │
│  │  │Viết lại  │    │      └─────┬──────┘ │  │  │
│  │  │câu hỏi   │    └───────────┬─────────┘  │  │
│  │  └──────────┘                │             │  │
│  │              ┌───────────────▼──────┐      │  │
│  │              │  Llama 3 (Groq API)  │      │  │
│  │              │  Trả lời cuối cùng   │      │  │
│  │              └──────────────────────┘      │  │
│  └───────────────────────────────────────────┘  │
│                       │                          │
│              ┌────────▼────────┐                 │
│              │    ChromaDB     │                 │
│              │  (Vector Store) │                 │
│              └─────────────────┘                 │
└─────────────────────────────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
RAG-Chatbot/
├── data/
│   ├── raw/                        # Tài liệu gốc (PDF, DOCX, TXT...)
│   ├── processed/                  # Dữ liệu đã qua tiền xử lý
│   └── vector_store/               # ChromaDB lưu trữ cục bộ
│
├── app/
│   ├── main.py                     # FastAPI Server + phục vụ Frontend
│   ├── api/
│   │   └── routes.py               # Định nghĩa API endpoints
│   ├── core/
│   │   ├── config.py               # Cấu hình ứng dụng
│   │   └── prompts.py              # Prompt templates (Luật sư + Viết lại câu)
│   ├── services/
│   │   ├── document_loader.py      # Đọc file PDF/DOCX/TXT
│   │   ├── text_splitter.py        # Chia nhỏ tài liệu thành chunks
│   │   ├── embedding.py            # Tích hợp model nhúng
│   │   ├── vector_db.py            # ChromaDB (có caching RAM)
│   │   ├── retriever.py            # Vector search đơn giản
│   │   ├── hybrid_retriever.py     # Hybrid Search + BM25 + Reranker
│   │   └── generator.py            # RAG Pipeline (Router + Memory + Generation)
│   └── ui/
│       ├── index.html              # Giao diện Chat chính (Dark mode)
│       └── chat_ui.py              # Giao diện Streamlit (bản đơn giản)
│
├── tests/
│   ├── test_rag.py                 # Unit tests (Latency + Router)
│   └── evaluate_rag.py             # Đánh giá chất lượng RAG (LLM-as-Judge)
│
├── notebooks/                      # Jupyter notebooks thử nghiệm
├── .env                            # API Keys (GROQ_API_KEY)
├── .env.example                    # Mẫu file .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Hướng dẫn cài đặt

### 1. Clone và tạo môi trường

```bash
git clone <repo-url>
cd RAG-Chatbot

# Tạo môi trường ảo
python -m venv venv

# Kích hoạt (Windows)
.\venv\Scripts\activate

# Cài đặt thư viện
pip install -r requirements.txt
pip install pypdf docx2txt rank-bm25
```

### 2. Thiết lập API Key

Tạo file `.env` ở thư mục gốc:

```env
GROQ_API_KEY=gsk_your_api_key_here
```

> 🔑 Lấy API Key miễn phí tại: [console.groq.com](https://console.groq.com)

### 3. Chuẩn bị dữ liệu

Đặt file tài liệu vào `data/raw/` (VD: `Bo_luat_lao_dong_2019.docx`), sau đó chạy:

```bash
# Bước 1: Đọc tài liệu
python app/services/document_loader.py

# Bước 2: Băm nhỏ tài liệu
python app/services/text_splitter.py

# Bước 3: Tạo Vector Database
python app/services/vector_db.py
```

### 4. Khởi động ứng dụng

```bash
# Chạy FastAPI Server + Frontend
uvicorn app.main:app --reload
```

Truy cập:
- 🌐 **Giao diện Chat**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 📄 **Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🧪 Kiểm thử

```bash
# Chạy Unit Tests (Latency + Router)
python tests/test_rag.py

# Chạy đánh giá chất lượng RAG tự động
python tests/evaluate_rag.py
```

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|------------|-----------|
| LLM | Llama 3.1 8B / Llama 3.3 70B (via Groq) |
| Embedding | `keepitreal/vietnamese-sbert` (HuggingFace) |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML / CSS / Vanilla JS |
| Orchestration | LangChain |

---

## 📊 Đánh giá chất lượng

Hệ thống được đánh giá tự động bằng phương pháp **LLM-as-a-Judge** theo 3 tiêu chí:

| Tiêu chí | Ý nghĩa |
|----------|---------|
| **Faithfulness** | Bot không bịa đặt thông tin ngoài tài liệu |
| **Relevancy** | Câu trả lời đúng trọng tâm câu hỏi |
| **Correctness** | Câu trả lời khớp với đáp án chuẩn |

---

## 📜 License

Dự án được phát triển cho mục đích học tập và nghiên cứu.
