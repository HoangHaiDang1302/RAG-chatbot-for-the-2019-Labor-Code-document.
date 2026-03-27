import os
import re
import sys
import time
import unicodedata
from typing import List
from collections import OrderedDict

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from app.services.vector_db import get_vector_db

_cached_bm25 = None
_cached_chunks = None
_cached_cross_encoder = None
_retrieval_cache: OrderedDict[str, tuple[float, List[str]]] = OrderedDict()
_RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "900"))
_RETRIEVAL_CACHE_MAX_ITEMS = int(os.getenv("RETRIEVAL_CACHE_MAX_ITEMS", "1024"))
_retrieval_metrics = {
    "retrieval_cache_hit": 0,
    "retrieval_cache_miss": 0,
}

_TOKEN_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "và",
    "là",
    "của",
    "theo",
    "các",
    "những",
    "một",
    "như",
    "cho",
    "trong",
    "với",
    "khi",
    "nào",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text.lower())
    text = _TOKEN_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _make_retrieval_cache_key(query: str, top_k: int) -> str:
    return f"{normalize_text(query)}|k={top_k}"


def _get_cached_retrieval(cache_key: str) -> List[str] | None:
    entry = _retrieval_cache.get(cache_key)
    if entry is None:
        return None

    expires_at, results = entry
    if time.time() > expires_at:
        _retrieval_cache.pop(cache_key, None)
        return None

    _retrieval_cache.move_to_end(cache_key)
    return results


def _set_cached_retrieval(cache_key: str, results: List[str]) -> None:
    _retrieval_cache[cache_key] = (time.time() + _RETRIEVAL_CACHE_TTL_SECONDS, list(results))
    _retrieval_cache.move_to_end(cache_key)

    while len(_retrieval_cache) > _RETRIEVAL_CACHE_MAX_ITEMS:
        _retrieval_cache.popitem(last=False)


def get_retrieval_runtime_metrics() -> dict:
    return dict(_retrieval_metrics)


def tokenize_text(text: str) -> List[str]:
    normalized = normalize_text(text)
    tokens = [token for token in normalized.split() if token and token not in _STOPWORDS]
    return tokens


def _build_bm25_index():
    global _cached_bm25, _cached_chunks

    if _cached_bm25 is not None:
        return _cached_bm25, _cached_chunks

    print("⏳ Đang xây chỉ mục BM25 (1 lần duy nhất)...")
    db = get_vector_db()
    all_data = db.get()
    all_texts = all_data["documents"]
    all_metadatas = all_data["metadatas"]

    tokenized_corpus = [tokenize_text(doc) for doc in all_texts]
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
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [(query, passage[:500]) for passage in candidates]
    scores = model.predict(pairs)

    scored = list(zip(scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [text for score, text in scored]


def vector_search(query: str, top_k: int = 3) -> List[str]:
    db = get_vector_db()
    results = db.similarity_search(query, k=top_k)
    return [doc.page_content for doc in results]


def bm25_search(query: str, top_k: int = 3) -> List[str]:
    bm25, chunks = _build_bm25_index()
    tokenized_query = tokenize_text(query)
    if not tokenized_query:
        return []

    bm25_scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
    return [chunks[i][0] for i in top_indices]


def _reciprocal_rank_fusion(vector_texts: List[str], bm25_texts: List[str], top_k: int) -> List[str]:
    scores = {}
    for rank, text in enumerate(vector_texts):
        scores[text] = scores.get(text, 0.0) + 1.0 / (rank + 1)
    for rank, text in enumerate(bm25_texts):
        scores[text] = scores.get(text, 0.0) + 1.0 / (rank + 1)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [text for text, score in ranked[:top_k]]


def hybrid_search(query: str, top_k: int = 3) -> List[str]:
    cache_key = _make_retrieval_cache_key(query, top_k)
    cached = _get_cached_retrieval(cache_key)
    if cached is not None:
        _retrieval_metrics["retrieval_cache_hit"] += 1
        return list(cached)
    _retrieval_metrics["retrieval_cache_miss"] += 1

    vector_texts = vector_search(query, top_k=top_k + 3)
    bm25_texts = bm25_search(query, top_k=top_k + 3)

    fused_candidates = _reciprocal_rank_fusion(vector_texts, bm25_texts, top_k=top_k + 5)
    reranked = rerank_cross_encoder(query, fused_candidates)
    final_results = reranked[:top_k]
    _set_cached_retrieval(cache_key, final_results)
    return final_results


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
