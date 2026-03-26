import os
import sys
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from app.services.vector_db import get_vector_db
from app.core.prompts import contextualize_q_prompt, qa_prompt
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Tiêu chuẩn màng lọc của lính gác cửa (Router)
router_prompt_template = """
Bạn là một lính gác cửa phân loại câu hỏi (Router). Nhiệm vụ độc nhất của bạn:
Kiểm tra CÂU HỎI MỚI của người dùng có chứa các đại từ nhân xưng, từ nối mập mờ (ví dụ: "vậy họ có được", "khi đó", "còn nghề đó", "thì sao", "đại loại vậy") khiến máy bắt buộc phải lật LỊCH SỬ CHAT VỪA RỒI lên đọc để hiểu hay không?

LỊCH SỬ CHAT THEO SAU:
{chat_history}

CÂU HỎI MỚI:
{input}

XEM XÉT VÀ TRẢ VỀ:
- Chữ "YES": Nếu câu mới mập mờ và giấu thông tin ở quá khứ.
- Chữ "NO": Nếu câu hỏi mới hoàn toàn rành mạch, có đọc riêng 1 mình nó ngta vẫn hiểu được.
"""

router_prompt = ChatPromptTemplate.from_template(router_prompt_template)

def generate_answer(query: str, chat_history: list = []):
    """
    RAG BẬC THẦY LATENCY VIP:
    Tự code lại luồng rẽ nhánh như hệ thống chuyên nghiệp của Silicon Valley!
    """
    try:
        if not os.environ.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY") == "your_groq_api_key_here":
            return "LỖI: Chưa có chìa khóa GROQ_API_KEY"
            
        # Model 1: Não Bự, Uyên Bác, Chậm Rãi (Chuyên viết luật)
        llm_tra_loi = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        # Model 2: Não Siêu Nhỏ Bằng Đốt Ngón Tay, Chạy Vi Tốc (Chỉ tốn 0.1 giây để gác cửa)
        llm_router_sieu_nhanh = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
        
        cau_hoi_chinh_xac = query
        
        # -------------------------------------------------------------
        # CÔNG ĐOẠN 1: MÀNG LỌC ROUTER (TRUY XÉT CÓ NÊN DỊCH LỊCH SỬ KHÔNG)
        # -------------------------------------------------------------
        if len(chat_history) > 0:
            print("\n🔍 Đang gọi Trọng tài siêu nhí (Model 8B) xét xử xem có nên đọc lại lịch sử không...")
            
            # Ép kết quả trả về đúng YES/NO
            def fix_router_output(msg):
                return "YES" if "yes" in msg.content.lower() else "NO"
            
            may_soi_xet = router_prompt | llm_router_sieu_nhanh
            router_decision = fix_router_output(may_soi_xet.invoke({
                "input": query, 
                "chat_history": "\n".join([m.content for m in chat_history[-2:]]) # Ép nó chỉ cần check 2 câu gần nhất cho siêu nhanh!
            }))
            
            if router_decision == "YES":
                print("➡️ Lính gác gõ búa: CÂU MẬP MỜ! Phải bắt Giám đốc (70B) thức dậy dịch lại câu!")
                may_viet_lai = contextualize_q_prompt | llm_tra_loi
                ket_qua_viet_lai = may_viet_lai.invoke({
                    "input": query,
                    "chat_history": chat_history
                })
                cau_hoi_chinh_xac = ket_qua_viet_lai.content
                print(f"👉 Giám đốc dịch lại thành: '{cau_hoi_chinh_xac}'")
            else:
                print("➡️ Lính gác gõ búa: CÂU HỎI MỚI TINH KHÔNG ÁM CHỈ! >>> BỎ QUA KHÂU DỊCH, TIẾT KIỆM TIỀN & THỜI GIAN NHÁ!!!")

        # -------------------------------------------------------------
        # CÔNG ĐOẠN 2: HYBRID SEARCH + RERANKER (Nâng cấp từ Vector thuần)
        # -------------------------------------------------------------
        print("\n📚 Đang chạy Hybrid Search (Vector + BM25 + Reranker)...")
        from app.services.hybrid_retriever import hybrid_search
        cac_mau_luat = hybrid_search(cau_hoi_chinh_xac, top_k=3)
        context = "\n\n---\n\n".join(cac_mau_luat)
        
        may_tra_loi = qa_prompt | llm_tra_loi
        response = may_tra_loi.invoke({
            "input": query,
            "context": context,
            "chat_history": chat_history
        })
        
        return response.content

    except Exception as e:
        return f"Sự cố hệ thống Router: {str(e)}"

if __name__ == "__main__":
    lich_su_chat = [
        HumanMessage(content="Lao động nữ nghỉ thai sản được bao nhiêu tháng?"),
        AIMessage(content="Lao động nữ được nghỉ trước và sau sinh trọn 6 tháng."),
    ]
    
    # Kịch bản 1: Mập mờ (Sẽ tốn thời gian dịch)
    cau_hoi_moi_1 = "Trong thời gian đó, họ có được hưởng nguyên lương không?"
    print(f"\n[Test Case 1] Hỏi cắc cớ: '{cau_hoi_moi_1}'")
    generate_answer(cau_hoi_moi_1, chat_history=lich_su_chat)
    
    # Kịch bản 2: Hỏi gắt 1 chủ đề biệt lập tươi rói (Sẽ bỏ qua khâu dịch, phóng thẳng)
    cau_hoi_moi_2 = "Phụ cấp ăn ca hiện nay là bao nhiêu tiền?"
    print(f"\n[Test Case 2] Hỏi chuyển đề tài: '{cau_hoi_moi_2}'")
    generate_answer(cau_hoi_moi_2, chat_history=lich_su_chat)

