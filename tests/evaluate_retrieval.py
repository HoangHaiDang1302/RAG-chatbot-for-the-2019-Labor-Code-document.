import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.hybrid_retriever import bm25_search, hybrid_search, normalize_text, vector_search


@dataclass
class RetrievalCase:
    query: str
    required_terms: List[str]


TEST_CASES = [
    RetrievalCase(
        query="Thời gian thử việc tối đa là bao lâu?",
        required_terms=["không quá 180 ngày", "không quá 60 ngày", "không quá 30 ngày", "không quá 6 ngày"],
    ),
    RetrievalCase(
        query="Lao động nữ nghỉ thai sản được bao nhiêu tháng?",
        required_terms=["6 tháng", "thai sản"],
    ),
    RetrievalCase(
        query="Tiền lương làm thêm giờ vào ban đêm được tính như thế nào?",
        required_terms=["150%", "200%", "300%", "20%", "30%"],
    ),
    RetrievalCase(
        query="Người sử dụng lao động phải báo trước bao nhiêu ngày khi đơn phương chấm dứt hợp đồng lao động?",
        required_terms=["45 ngày", "30 ngày"],
    ),
    RetrievalCase(
        query="Người lao động được nghỉ bao nhiêu ngày lễ, tết trong năm?",
        required_terms=["11 ngày"],
    ),
]


def _is_relevant(text: str, required_terms: List[str]) -> List[str]:
    normalized = normalize_text(text)
    matched = []
    for term in required_terms:
        if normalize_text(term) in normalized:
            matched.append(term)
    return matched


def _precision_at_k(results: List[str], required_terms: List[str]) -> float:
    if not results:
        return 0.0
    relevant_count = sum(1 for text in results if _is_relevant(text, required_terms))
    return round(relevant_count / len(results), 3)


def _recall_at_k(results: List[str], required_terms: List[str]) -> float:
    if not required_terms:
        return 0.0
    covered_terms = set()
    for text in results:
        covered_terms.update(_is_relevant(text, required_terms))
    return round(len(covered_terms) / len(required_terms), 3)


def _mrr(results: List[str], required_terms: List[str]) -> float:
    for idx, text in enumerate(results, start=1):
        if _is_relevant(text, required_terms):
            return round(1.0 / idx, 3)
    return 0.0


def _benchmark(name: str, search_fn: Callable[[str, int], List[str]], top_k: int = 3) -> Dict[str, float]:
    precision_scores = []
    recall_scores = []
    mrr_scores = []

    for case in TEST_CASES:
        results = search_fn(case.query, top_k=top_k)
        precision_scores.append(_precision_at_k(results, case.required_terms))
        recall_scores.append(_recall_at_k(results, case.required_terms))
        mrr_scores.append(_mrr(results, case.required_terms))

    return {
        "name": name,
        "precision@k": round(sum(precision_scores) / len(precision_scores), 3),
        "recall@k": round(sum(recall_scores) / len(recall_scores), 3),
        "mrr": round(sum(mrr_scores) / len(mrr_scores), 3),
    }


def run_benchmark():
    benches = [
        _benchmark("vector", vector_search),
        _benchmark("bm25", bm25_search),
        _benchmark("hybrid", hybrid_search),
    ]

    print("=" * 72)
    print("BENCHMARK RETRIEVAL")
    print("=" * 72)
    for item in benches:
        print(
            f"{item['name']:>8} | precision@k={item['precision@k']:.3f} | "
            f"recall@k={item['recall@k']:.3f} | mrr={item['mrr']:.3f}"
        )
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
