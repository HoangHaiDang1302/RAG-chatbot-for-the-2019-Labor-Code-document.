import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma

# Đường dẫn DB
DB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "vector_store")

# Cache trên RAM
_cached_db = None
_cached_embeddings = None

def get_embedding_model():
    """
    Khởi tạo FastEmbed (Chạy Local, dùng chuẩn nén ONNX siêu nhẹ).
    Cold start: siêu tốc ~1-2 giây. Không giới hạn request API, miễn phí 100%.
    """
    global _cached_embeddings
    
    if _cached_embeddings is None:
        print("⚡ Khởi động FastEmbed Local (Sẽ mất chút thời gian tải Model lần đầu)...")
        # Sử dụng model đa ngôn ngữ tương thích FastEmbed (Mã số đã được sửa lại cho đúng Tên Model)
        _cached_embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        print("✅ Model ONNX đã nạp xong vào RAM!")
    return _cached_embeddings

def create_vector_db(chunks):
    """
    Xây mới kho VectorDB bằng FastEmbed.
    Chạy sức mạnh CPU nội bộ 100%, không bị giới hạn 100 req/phút của API nào hết!
    Nên giờ chúng ta không cần chia lô 80 và vã mồ hôi nằm chờ 60 giây như Google nữa!
    """
    embeddings = get_embedding_model()
    print(f"📦 Đang dịch 1 mạch (embed) {len(chunks)} chunks thành số (Vector) nhanh như chớp...")
    
    # Làm 1 lệnh nhét thẳng vô không cần chia lô
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    print(f"✅ Đã dập thành công vào kho VectorDB nội bộ!")
    return db

def get_vector_db():
    """Phục hồi DB từ ổ cứng (có cache RAM để chống load đi load lại nhiều lần)."""
    global _cached_db
    
    if _cached_db is None:
        print("⏳ Đang kết nối Vector DB Local...")
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
    print("🔄 TÁI SINH VÀ LÊN ĐỜI VECTOR DB LÊN CHUẨN FASTEMBED (ONNX)")
    print("="*50)
    
    # Xóa DB cũ để tránh xung đột mã số học sinh
    import shutil
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("🗑️ Đã cho Thùng rác DB cũ (vì không cùng ngôn ngữ Embedding)")
    
    docs = load_document(sample_file)
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=200)
    db = create_vector_db(chunks)
    
    print("\n✅ THÀNH CÔNG! Vector DB xịn xò chạy Offine Không Giới Hạn đã lên sàn!")
