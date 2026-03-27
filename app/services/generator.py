import os
import sys
import re
import time
import hashlib
import unicodedata
from typing import Any, List
from collections import OrderedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from app.core.prompts import contextualize_q_prompt, qa_prompt
from app.services.hybrid_retriever import hybrid_search

router_prompt_template = """
Bạn là bộ phân loại câu hỏi (Router).
Nhiệm vụ duy nhất của bạn là quyết định câu hỏi mới có cần dùng lịch sử hội thoại để hiểu đúng ý hay không.

Hãy trả về đúng một trong hai giá trị sau:
- YES: nếu câu hỏi mới có đại từ, từ nối, hoặc cách diễn đạt mơ hồ phụ thuộc vào ngữ cảnh trước đó.
- NO: nếu câu hỏi mới đã độc lập, rõ nghĩa, và có thể hiểu mà không cần xem lịch sử.

LỊCH SỬ HỘI THOẠI:
{chat_history}

CÂU HỎI MỚI:
{input}

Chỉ trả về YES hoặc NO, không giải thích thêm.
"""

router_prompt = ChatPromptTemplate.from_template(router_prompt_template)

_context_cache: OrderedDict[str, tuple[float, str, str]] = OrderedDict()
_CACHE_TTL_SECONDS = int(os.getenv("CONTEXT_CACHE_TTL_SECONDS", "900"))
_CACHE_MAX_ITEMS = int(os.getenv("CONTEXT_CACHE_MAX_ITEMS", "512"))
_answer_cache: OrderedDict[str, tuple[float, str]] = OrderedDict()
_ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "600"))
_ANSWER_CACHE_MAX_ITEMS = int(os.getenv("ANSWER_CACHE_MAX_ITEMS", "512"))
_ROUTER_YES_THRESHOLD = float(os.getenv("ROUTER_YES_THRESHOLD", "0.65"))
_ROUTER_NO_THRESHOLD = float(os.getenv("ROUTER_NO_THRESHOLD", "0.35"))
_ROUTER_RULE_ONLY = os.getenv("ROUTER_RULE_ONLY", "0") == "1"

_SPACE_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_COREF_HINTS = {
    "đó",
    "vậy",
    "này",
    "kia",
    "thế",
    "như vậy",
    "trong trường hợp đó",
    "ở trên",
    "bên trên",
    "phần trên",
    "điều đó",
    "khoản đó",
}
_LEADING_FOLLOWUP_HINTS = (
    "thế",
    "vậy",
    "còn",
    "như vậy",
    "trong trường hợp đó",
    "với trường hợp đó",
)
_metrics = {
    "answer_cache_hit": 0,
    "answer_cache_miss": 0,
    "context_cache_hit": 0,
    "context_cache_miss": 0,
    "router_rule_yes": 0,
    "router_rule_no": 0,
    "router_llm_fallback": 0,
}


def _normalize_for_cache(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", (text or "").strip().lower())
    return _SPACE_RE.sub(" ", normalized)


def _history_fingerprint(chat_history: List[Any], tail_size: int = 4) -> str:
    tail = chat_history[-tail_size:] if len(chat_history) > tail_size else chat_history
    lines = []
    for msg in tail:
        content = getattr(msg, "content", "")
        lines.append(_normalize_for_cache(content))
    joined = "\n".join(lines)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _tokenize_simple(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize_for_cache(text))


def _overlap_ratio(a_tokens: List[str], b_tokens: List[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a_set = set(a_tokens)
    b_set = set(b_tokens)
    if not a_set or not b_set:
        return 0.0
    return len(a_set & b_set) / max(1, len(a_set))


def _rule_router_score(query: str, chat_history: List[Any]) -> float:
    q_norm = _normalize_for_cache(query)
    q_tokens = _tokenize_simple(query)

    score = 0.0

    if any(hint in q_norm for hint in _COREF_HINTS):
        score += 0.55

    if any(q_norm.startswith(prefix) for prefix in _LEADING_FOLLOWUP_HINTS):
        score += 0.20

    if len(q_tokens) <= 8:
        score += 0.10

    last_user_msg = ""
    for msg in reversed(chat_history):
        if isinstance(msg, HumanMessage):
            last_user_msg = getattr(msg, "content", "") or ""
            break

    if last_user_msg:
        overlap = _overlap_ratio(q_tokens, _tokenize_simple(last_user_msg))
        if overlap >= 0.45:
            score += 0.10

    return min(1.0, max(0.0, score))


def _rule_router_decision(query: str, chat_history: List[Any]) -> str:
    score = _rule_router_score(query, chat_history)
    if score >= _ROUTER_YES_THRESHOLD:
        return "YES"
    if score <= _ROUTER_NO_THRESHOLD:
        return "NO"
    return "UNKNOWN"


def get_runtime_metrics() -> dict:
    return dict(_metrics)


def _make_context_cache_key(query: str, chat_history: List[Any]) -> str:
    q_norm = _normalize_for_cache(query)
    h_fingerprint = _history_fingerprint(chat_history)
    raw_key = f"{q_norm}|{h_fingerprint}"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def _make_answer_cache_key(query: str, chat_history: List[Any]) -> str:
    q_norm = _normalize_for_cache(query)
    h_fingerprint = _history_fingerprint(chat_history)
    raw_key = f"{q_norm}|{h_fingerprint}|answer"
    return hashlib.sha1(raw_key.encode("utf-8")).hexdigest()


def _get_cached_context(cache_key: str) -> tuple[str, str] | None:
    entry = _context_cache.get(cache_key)
    if entry is None:
        return None

    expires_at, router_decision, rewritten_query = entry
    if time.time() > expires_at:
        _context_cache.pop(cache_key, None)
        return None

    _context_cache.move_to_end(cache_key)
    return router_decision, rewritten_query


def _set_cached_context(cache_key: str, router_decision: str, rewritten_query: str) -> None:
    _context_cache[cache_key] = (time.time() + _CACHE_TTL_SECONDS, router_decision, rewritten_query)
    _context_cache.move_to_end(cache_key)

    while len(_context_cache) > _CACHE_MAX_ITEMS:
        _context_cache.popitem(last=False)


def _get_cached_answer(cache_key: str) -> str | None:
    entry = _answer_cache.get(cache_key)
    if entry is None:
        return None

    expires_at, answer = entry
    if time.time() > expires_at:
        _answer_cache.pop(cache_key, None)
        return None

    _answer_cache.move_to_end(cache_key)
    return answer


def _set_cached_answer(cache_key: str, answer: str) -> None:
    _answer_cache[cache_key] = (time.time() + _ANSWER_CACHE_TTL_SECONDS, answer)
    _answer_cache.move_to_end(cache_key)

    while len(_answer_cache) > _ANSWER_CACHE_MAX_ITEMS:
        _answer_cache.popitem(last=False)


def _serialize_history(chat_history: List[Any]) -> str:
    return "\n".join([m.content for m in chat_history[-2:]])


def generate_answer(query: str, chat_history: List[Any] | None = None):
    chat_history = chat_history or []

    try:
        if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
            return "Lỗi: Chưa có chìa khóa GROQ_API_KEY"

        answer_cache_key = _make_answer_cache_key(query, chat_history)
        cached_answer = _get_cached_answer(answer_cache_key)
        if cached_answer is not None:
            _metrics["answer_cache_hit"] += 1
            return cached_answer
        _metrics["answer_cache_miss"] += 1

        llm_tra_loi = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

        cau_hoi_chinh_xac = query

        if len(chat_history) > 0:
            def fix_router_output(msg):
                content = (msg.content or "").strip().upper()
                return "YES" if content.startswith("YES") else "NO"

            cache_key = _make_context_cache_key(query, chat_history)
            cached = _get_cached_context(cache_key)

            if cached is not None:
                _metrics["context_cache_hit"] += 1
                router_decision, cau_hoi_chinh_xac = cached
            else:
                _metrics["context_cache_miss"] += 1
                router_decision = _rule_router_decision(query, chat_history)

                if router_decision == "UNKNOWN":
                    if _ROUTER_RULE_ONLY:
                        router_decision = "NO"
                        _metrics["router_rule_no"] += 1
                    else:
                        _metrics["router_llm_fallback"] += 1
                        llm_router_sieu_nhanh = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
                        may_soi_xet = router_prompt | llm_router_sieu_nhanh
                        router_decision = fix_router_output(
                            may_soi_xet.invoke(
                                {
                                    "input": query,
                                    "chat_history": _serialize_history(chat_history),
                                }
                            )
                        )
                elif router_decision == "YES":
                    _metrics["router_rule_yes"] += 1
                else:
                    _metrics["router_rule_no"] += 1

                if router_decision == "YES":
                    may_viet_lai = contextualize_q_prompt | llm_tra_loi
                    ket_qua_viet_lai = may_viet_lai.invoke(
                        {
                            "input": query,
                            "chat_history": chat_history,
                        }
                    )
                    cau_hoi_chinh_xac = ket_qua_viet_lai.content

                _set_cached_context(cache_key, router_decision, cau_hoi_chinh_xac)

        cac_mau_luat = hybrid_search(cau_hoi_chinh_xac, top_k=3)
        context = "\n\n---\n\n".join(cac_mau_luat)

        may_tra_loi = qa_prompt | llm_tra_loi
        response = may_tra_loi.invoke(
            {
                "input": query,
                "context": context,
                "chat_history": chat_history,
            }
        )

        final_answer = response.content
        _set_cached_answer(answer_cache_key, final_answer)
        return final_answer

    except Exception as e:
        return f"Sự cố hệ thống Router: {str(e)}"


if __name__ == "__main__":
    lich_su_chat = [
        HumanMessage(content="Lao động nữ nghỉ thai sản được bao nhiêu tháng?"),
        AIMessage(content="Lao động nữ được nghỉ trước và sau sinh trọn 6 tháng."),
    ]

    cau_hoi_moi_1 = "Trong thời gian đó, họ có được hưởng nguyên lương không?"
    print(f"\n[Test Case 1] Hỏi cắc cớ: '{cau_hoi_moi_1}'")
    print(generate_answer(cau_hoi_moi_1, chat_history=lich_su_chat))

    cau_hoi_moi_2 = "Phụ cấp ăn ca hiện nay là bao nhiêu tiền?"
    print(f"\n[Test Case 2] Hỏi chuyển đề tài: '{cau_hoi_moi_2}'")
    print(generate_answer(cau_hoi_moi_2, chat_history=lich_su_chat))
