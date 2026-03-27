import os
import sys
from typing import List

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from app.services.vector_db import get_vector_db

_cached_bm25 = None
_cached_chunks = None
_cached_cross_encoder = None


def _build_bm25_index():
    global _cached_bm25, _cached_chunks

    if _cached_bm25 is not None:
        return _cached_bm25, _cached_chunks

    print("⏳ Đang xây bảng tra cứu BM25 (1 lần duy nhất)...")
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
    global _cached_cross_encoder

    if _cached_cross_encoder is None:
        print("⏳ Đang tải Cross-Encoder lên RAM (1 lần duy nhất, ~50MB)...")
        from sentence_transformers import CrossEncoder

        _cached_cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("✅ Cross-Encoder sẵn sàng!")

    return _cached_cross_encoder


def rerank_cross_encoder(query: str, candidates: List[str]) -> List[str]:
    model = _get_cross_encoder()
    pairs = [(query, passage[:500]) for passage in candidates]
    scores = model.predict(pairs)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [text for score, text in scored]


def _hybrid_search_items(query: str, top_k: int = 3) -> List[str]:
    db = get_vector_db()
    vector_results = db.similarity_search(query, k=top_k + 2)
    vector_texts = [doc.page_content for doc in vector_results]

    bm25, chunks = _build_bm25_index()
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[: top_k + 2]
    bm25_texts = [chunks[i][0] for i in top_indices]

    seen = set()
    merged_texts: List[str] = []
    for text in vector_texts + bm25_texts:
        fingerprint = text[:120]
        if fingerprint not in seen:
            seen.add(fingerprint)
            merged_texts.append(text)

    reranked_texts = rerank_cross_encoder(query, merged_texts)
    return reranked_texts[:top_k]


def hybrid_search(query: str, top_k: int = 3) -> List[str]:
    return _hybrid_search_items(query, top_k=top_k)


if __name__ == "__main__":
    import time

    print("🔥 Warm-up: Tải model lên RAM...")
    _get_cross_encoder()
    _build_bm25_index()
    get_vector_db().similarity_search("test", k=1)
    print("✅ Sẵn sàng!\n")

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
