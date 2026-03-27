import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Đường dẫn DB
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_store")

# Cache trên RAM
_cached_db = None
_cached_embeddings = None

def get_embedding_model():
    """
    Khởi tạo Google Gemini Embedding (Cloud API).
    Cold start: 0 giây! Model nằm sẵn trên server Google.
    """
    global _cached_embeddings
    
    if _cached_embeddings is None:
        print("⚡ Kết nối Google Gemini Embedding API (0 giây cold start)...")
        _cached_embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.environ.get("GOOGLE_API_KEY")
        )
    return _cached_embeddings

def create_vector_db(chunks):
    """
    Xây mới kho VectorDB bằng Google Embedding.
    Chia nhỏ thành từng lô 80 chunks để không vượt giới hạn 100 req/phút của Google Free Tier.
    """
    import time
    
    embeddings = get_embedding_model()
    BATCH_SIZE = 80
    total = len(chunks)
    
    print(f"Đang tạo Vector DB với {total} chunks (chia thành lô {BATCH_SIZE})...")
    
    db = None
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"   📦 Lô {batch_num}/{total_batches}: đang embed {len(batch)} chunks...")
        
        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=DB_DIR
            )
        else:
            db.add_documents(batch)
        
        # Nghỉ 60 giây nếu còn lô tiếp theo (chờ Google reset quota)
        if i + BATCH_SIZE < total:
            print(f"   ⏳ Chờ 60 giây để Google reset quota...")
            time.sleep(60)
    
    print(f"✅ Đã lưu toàn bộ {total} chunks vào DB!")
    return db

def get_vector_db():
    """Phục hồi DB từ ổ cứng (có cache RAM)."""
    global _cached_db
    
    if _cached_db is None:
        print("⏳ Đang kết nối Vector DB...")
        embeddings = get_embedding_model()
        _cached_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    return _cached_db

# Rebuild DB khi chạy trực tiếp
if __name__ == "__main__":
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.services.document_loader import load_document
    from app.services.text_splitter import split_documents
    
    sample_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Bo_luat_lao_dong_2019.docx")
    
    print("="*50)
    print("🔄 XÂY LẠI VECTOR DB VỚI GOOGLE EMBEDDING")
    print("="*50)
    
    # Xóa DB cũ
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("🗑️ Đã xóa DB cũ (embedding cũ không tương thích)")
    
    docs = load_document(sample_file)
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    db = create_vector_db(chunks)
    
    print("\n✅ XONG! Vector DB mới đã sẵn sàng với Google Embedding!")
