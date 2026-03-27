import os
import sys
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from app.services.vector_db import get_vector_db

# Biến Toàn Cục (Cache trên RAM)
_cached_bm25 = None
_cached_chunks = None
_cached_cross_encoder = None

def _build_bm25_index():
    """Xây Bảng Tra Cứu Từ Khóa BM25."""
    global _cached_bm25, _cached_chunks
    
    if _cached_bm25 is not None:
        return _cached_bm25, _cached_chunks
    
    print("⏳ Đang xây Bảng Tra Cứu BM25 (1 lần duy nhất)...")
    db = get_vector_db()
    all_data = db.get()
    all_texts = all_data["documents"]
    all_metadatas = all_data["metadatas"]
    
    tokenized_corpus = [doc.lower().split() for doc in all_texts]
    _cached_bm25 = BM25Okapi(tokenized_corpus)
    _cached_chunks = list(zip(all_texts, all_metadatas))
    
    print(f"✅ BM25 sẵn sàng với {len(all_texts)} chunks.")
    return _cached_bm25, _cached_chunks

def _get_cross_encoder():
    """Tải Cross-Encoder model lên RAM (1 lần duy nhất)."""
    global _cached_cross_encoder
    
    if _cached_cross_encoder is None:
        print("⏳ Đang tải Cross-Encoder lên RAM (1 lần duy nhất, ~50MB)...")
        from sentence_transformers import CrossEncoder
        _cached_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Cross-Encoder sẵn sàng!")
    
    return _cached_cross_encoder

def rerank_cross_encoder(query: str, candidates: list) -> list:
    """
    Cross-Encoder Reranker: Model cục bộ chấm điểm tất cả cùng lúc (batch).
    Nhanh (~0.2 giây) và chính xác cao.
    """
    model = _get_cross_encoder()
    pairs = [(query, passage[:500]) for passage in candidates]
    scores = model.predict(pairs)
    
    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [text for score, text in scored]

def hybrid_search(query: str, top_k: int = 3):
    """
    Hybrid Search: Vector + BM25 + Cross-Encoder Reranker.
    """
    # ---- NHÁNH 1: VECTOR SEARCH ----
    db = get_vector_db()
    vector_results = db.similarity_search(query, k=top_k + 2)
    vector_texts = [doc.page_content for doc in vector_results]
    
    # ---- NHÁNH 2: BM25 SEARCH ----
    bm25, chunks = _build_bm25_index()
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k + 2]
    bm25_texts = [chunks[i][0] for i in top_indices]
    
    # ---- TRỘN & KHỬ TRÙNG ----
    seen = set()
    merged = []
    for text in vector_texts + bm25_texts:
        fp = text[:100]
        if fp not in seen:
            seen.add(fp)
            merged.append(text)
    
    # ---- CROSS-ENCODER RERANK ----
    reranked = rerank_cross_encoder(query, merged)
    
    return reranked[:top_k]

# ---- TEST ----
if __name__ == "__main__":
    import time
    
    # Warm-up
    print("🔥 Warm-up: Tải model lên RAM...")
    _get_cross_encoder()
    _build_bm25_index()
    get_vector_db().similarity_search("test", k=1)
    print("✅ Sẵn sàng!\n")
    
    # Benchmark
    queries = [
        "Phụ nữ mang thai được nghỉ bao lâu?",
        "Điều 98 quy định gì?",
        "Tiền lương làm thêm giờ ban đêm tính thế nào?",
    ]
    
    for q in queries:
        start = time.time()
        results = hybrid_search(q, top_k=3)
        elapsed = time.time() - start
        print(f"⏱️ {elapsed:.2f}s | '{q}'")
        print(f"   → {results[0][:80]}...\n")
