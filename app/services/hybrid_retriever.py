import os
import sys
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from app.services.vector_db import get_vector_db

# Biến Toàn Cục: Ghim bộ máy BM25 vào RAM (tránh tải đi tải lại nhiều lần)
_cached_bm25 = None
_cached_chunks = None

def _build_bm25_index():
    """
    Xây dựng Chỉ Mục BM25 (Bảng tra cứu Từ Khóa).
    Lấy toàn bộ chunks trong ChromaDB ra, tách từng chữ, nạp vào BM25.
    """
    global _cached_bm25, _cached_chunks
    
    if _cached_bm25 is not None:
        return _cached_bm25, _cached_chunks
    
    print("⏳ Đang xây Bảng Tra Cứu Từ Khóa BM25 (Chỉ tốn 1 lần)...")
    db = get_vector_db()
    
    # Lôi toàn bộ tài liệu thô từ kho ChromaDB ra
    all_data = db.get()
    all_texts = all_data["documents"]   # Danh sách các đoạn văn bản
    all_metadatas = all_data["metadatas"]  # Metadata đi kèm mỗi đoạn
    
    # BM25 cần mỗi đoạn văn ở dạng danh sách các từ (tokenized)
    tokenized_corpus = [doc.lower().split() for doc in all_texts]
    
    _cached_bm25 = BM25Okapi(tokenized_corpus)
    _cached_chunks = list(zip(all_texts, all_metadatas))
    
    print(f"✅ Đã xây xong Bảng BM25 với {len(all_texts)} khối văn bản.")
    return _cached_bm25, _cached_chunks

def hybrid_search(query: str, top_k: int = 3):
    """
    TÌM KIẾM LAI (Hybrid Search):
    - Nhánh 1: Vector Search (Tìm theo Ý Nghĩa)
    - Nhánh 2: BM25 Search (Tìm theo Từ Khóa Chính Xác)
    - Bộ lọc: Trộn lại -> Loại trùng -> Dùng LLM nhỏ chấm điểm (Reranker)
    """
    # ---- NHÁNH 1: VECTOR SEARCH (Tìm theo toạ độ Ý nghĩa) ----
    db = get_vector_db()
    vector_results = db.similarity_search(query, k=top_k)
    vector_texts = [doc.page_content for doc in vector_results]
    
    print(f"\n🔵 Vector Search lôi được {len(vector_texts)} mẩu theo ý nghĩa.")
    
    # ---- NHÁNH 2: BM25 SEARCH (Đếm Từ Khóa Chính Xác) ----
    bm25, chunks = _build_bm25_index()
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Lấy chỉ số (index) của top_k đoạn có điểm BM25 cao nhất
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    bm25_texts = [chunks[i][0] for i in top_bm25_indices]
    
    print(f"🟡 BM25 Search lôi được {len(bm25_texts)} mẩu theo từ khóa.")
    
    # ---- TRỘN & KHỬ TRÙNG ----
    seen = set()
    merged = []
    for text in vector_texts + bm25_texts:
        # Dùng 100 ký tự đầu làm "vân tay" nhận diện trùng lặp
        fingerprint = text[:100]
        if fingerprint not in seen:
            seen.add(fingerprint)
            merged.append(text)
    
    print(f"🟢 Sau khi trộn và khử trùng: còn {len(merged)} mẩu độc nhất.")
    
    # ---- RERANKER (Chấm Điểm Lại bằng LLM siêu nhỏ) ----
    reranked = rerank_with_llm(query, merged, top_k=top_k)
    
    return reranked

def rerank_with_llm(query: str, candidates: list, top_k: int = 3):
    """
    RERANKER: Dùng LLM siêu nhỏ để chấm điểm lại từng đoạn văn bản.
    Đoạn nào liên quan nhất với câu hỏi sẽ được chọn.
    """
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    rerank_prompt = ChatPromptTemplate.from_template("""
Hãy chấm điểm từ 0 đến 10 mức độ liên quan giữa CÂU HỎI và ĐOẠN VĂN BẢN dưới đây.
Chỉ trả về DUY NHẤT MỘT CON SỐ (ví dụ: 8). Không giải thích.

CÂU HỎI: {question}

ĐOẠN VĂN BẢN: {passage}

ĐIỂM:""")
    
    scored = []
    print(f"\n📊 Reranker đang chấm điểm {len(candidates)} ứng viên...")
    
    for i, passage in enumerate(candidates):
        try:
            chain = rerank_prompt | llm
            result = chain.invoke({"question": query, "passage": passage[:500]})
            # Trích con số điểm từ phản hồi
            score_text = result.content.strip()
            score = float(''.join(c for c in score_text if c.isdigit() or c == '.') or '0')
        except Exception:
            score = 0
        
        scored.append((score, passage))
        print(f"   Ứng viên {i+1}: {score}/10 điểm")
    
    # Xếp theo điểm giảm dần, lấy top_k đoạn ngon nhất
    scored.sort(key=lambda x: x[0], reverse=True)
    winners = [text for score, text in scored[:top_k]]
    
    print(f"🏆 Reranker đã chọn ra {len(winners)} đoạn chất lượng nhất!")
    return winners

# ---- TEST THỬ ----
if __name__ == "__main__":
    print("="*60)
    print("TEST 1: Câu hỏi theo Ý Nghĩa (Vector sẽ giỏi)")
    print("="*60)
    results_1 = hybrid_search("Phụ nữ mang thai được nghỉ bao lâu?", top_k=3)
    for i, r in enumerate(results_1):
        print(f"\n--- Kết quả {i+1} ---")
        print(r[:200])
    
    print("\n" + "="*60)
    print("TEST 2: Câu hỏi theo Từ Khóa Chính Xác (BM25 sẽ giỏi)")
    print("="*60)
    results_2 = hybrid_search("Điều 98", top_k=3)
    for i, r in enumerate(results_2):
        print(f"\n--- Kết quả {i+1} ---")
        print(r[:200])
