import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Đường dẫn DB
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_store")

# Biến Toàn Cục để lưu trữ trên RAM (KHÔNG PHẢI TẢI LẠI NHIỀU LẦN)
_cached_db = None

def get_embedding_model():
    """ Khởi tạo mô hình dịch tiếng Việt thành Số """
    embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
    return embeddings

def create_vector_db(chunks):
    """
    Xây mới kho VectorDB
    """
    embeddings = get_embedding_model()
    print("Đang tạo Vector DB và cất vào kho, vui lòng đợi...")
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"✅ Tuyệt vời! Đã lưu toàn bộ khối văn bản vào DB cục bộ.")
    return db

def get_vector_db():
    """
    Phục hồi DB từ ổ đĩa cứng.
    ĐIỂM NÂNG CẤP TỐI ƯU TỐC ĐỘ: Lưu luôn DB trên RAM vào biến _cached_db
    """
    global _cached_db
    
    # Nếu trong RAM chưa có đồ, bắt buộc phải tải lên từ Ổ Cứng
    if _cached_db is None:
        print("⏳ Đang tải mô hình Nhúng Embedding lên RAM Máy (Chỉ tốn đúng 1 lần đầu tiên)...")
        embeddings = get_embedding_model()
        _cached_db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    return _cached_db
