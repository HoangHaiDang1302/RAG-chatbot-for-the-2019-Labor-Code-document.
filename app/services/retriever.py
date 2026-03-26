import os
import sys

# Khai báo đường dẫn để import từ các file tự viết trước đó
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.services.vector_db import get_vector_db

def search_documents(query: str, top_k: int = 3):
    """
    Hàm này đại diện chữ R (Retrieval) trong RAG. 
    Nó nhận câu hỏi Tiếng Việt -> Biến câu thành Vector Số ->
    Lôi trong Kho (ChromaDB) ra "top_k" mẩu bánh mì (chunks) gần nghĩa nhất.
    """
    print(f"\n🔍 Đang lặn xuống Kho DB để lục lọi ý nghĩa câu: '{query}' ...")
    db = get_vector_db()
    
    # Dùng kỹ thuật truy xuất "Similarity Search" (Khoảng cách điểm ảnh / Cùng Cosine độ lớn)
    # nôm na là tìm ra độ tương đồng giữa các chuỗi số tọa độ. 
    results = db.similarity_search(query, k=top_k)
    return results

if __name__ == "__main__":
    # Bây giờ, thay vì đọc cả bộ luật, bạn là 1 người hỏi bài
    cau_hoi = "Làm thêm giờ vào ngày lễ tết thì được trả lương thế nào?"
    
    try:
        # Nhờ hàm này chạy vô kho lôi ra 3 đoạn (khối) có ý nghĩa khớp nhất
        tai_lieu_tim_thay = search_documents(cau_hoi, top_k=3)
        
        print(f"\n✅ Tada! Đã tìm thấy {len(tai_lieu_tim_thay)} mẩu luật liên quan (Retrieval). Đây là kết quả thô:\n")
        
        # Lặp để in ra xem kho tìm thấy đoạn văn nào trong Bộ Luật 2019 nhé
        for i, doc in enumerate(tai_lieu_tim_thay):
            print(f"--- ĐIỀU LUẬT KHỚP Ý SỐ {i + 1} ---")
            print(doc.page_content) # text văn bản thô
            print("-" * 50 + "\n")
            
    except Exception as e:
        print(f"Lỗi: {e}")
