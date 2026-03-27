import os
import sys
from typing import Any, List

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


def _serialize_history(chat_history: List[Any]) -> str:
    return "\n".join([m.content for m in chat_history[-2:]])


def generate_answer(query: str, chat_history: List[Any] | None = None):
    chat_history = chat_history or []

    try:
        if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
            return "Lỗi: Chưa có chìa khóa GROQ_API_KEY"

        llm_tra_loi = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        llm_router_sieu_nhanh = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

        cau_hoi_chinh_xac = query

        if len(chat_history) > 0:
            def fix_router_output(msg):
                content = (msg.content or "").strip().upper()
                return "YES" if content.startswith("YES") else "NO"

            may_soi_xet = router_prompt | llm_router_sieu_nhanh
            router_decision = fix_router_output(
                may_soi_xet.invoke(
                    {
                        "input": query,
                        "chat_history": _serialize_history(chat_history),
                    }
                )
            )

            if router_decision == "YES":
                may_viet_lai = contextualize_q_prompt | llm_tra_loi
                ket_qua_viet_lai = may_viet_lai.invoke(
                    {
                        "input": query,
                        "chat_history": chat_history,
                    }
                )
                cau_hoi_chinh_xac = ket_qua_viet_lai.content

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

        return response.content

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
