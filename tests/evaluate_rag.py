import os
import sys
import json
from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from app.services.hybrid_retriever import hybrid_search
from app.services.generator import generate_answer

# ============================================================
# BỘ ĐỀ THI TỰ ĐỘNG (Test Suite)
# Mỗi câu có: câu hỏi + đáp án đúng (ground truth) để so sánh
# ============================================================
TEST_CASES = [
    {
        "question": "Người lao động được nghỉ bao nhiêu ngày lễ, tết trong năm?",
        "ground_truth": "11 ngày nghỉ lễ, tết trong năm"
    },
    {
        "question": "Thời gian thử việc tối đa là bao lâu?",
        "ground_truth": "Không quá 180 ngày đối với công việc trình độ cao đẳng trở lên, không quá 60 ngày với trung cấp, không quá 30 ngày với công việc khác, không quá 6 ngày với công việc đơn giản"
    },
    {
        "question": "Lao động nữ nghỉ thai sản được bao nhiêu tháng?",
        "ground_truth": "Lao động nữ được nghỉ thai sản trước và sau sinh là 6 tháng"
    },
    {
        "question": "Người sử dụng lao động phải báo trước bao nhiêu ngày khi đơn phương chấm dứt hợp đồng lao động?",
        "ground_truth": "Ít nhất 45 ngày với hợp đồng không xác định thời hạn, 30 ngày với hợp đồng xác định thời hạn"
    },
    {
        "question": "Tiền lương làm thêm giờ vào ban đêm được tính như thế nào?",
        "ground_truth": "Ít nhất bằng 150% vào ngày thường, 200% vào ngày nghỉ hằng tuần, 300% vào ngày lễ tết, cộng thêm 20% tiền lương ban đêm và 30% tiền lương làm thêm giờ ban đêm"
    },
]

# ============================================================
# BỘ GIÁM KHẢO AI: Chấm điểm bằng LLM
# ============================================================
def judge_faithfulness(question: str, answer: str, context: str, llm) -> float:
    """
    TIÊU CHÍ 1: TRUNG THỰC (Faithfulness)
    Bot có bịa ra thông tin nào KHÔNG có trong tài liệu không?
    """
    prompt = ChatPromptTemplate.from_template("""
Bạn là Giám khảo AI. Hãy kiểm tra xem CÂU TRẢ LỜI có chứa thông tin nào KHÔNG CÓ trong TÀI LIỆU không.
Chấm điểm từ 0.0 đến 1.0:
- 1.0 = Toàn bộ thông tin trong câu trả lời đều có trong tài liệu (Trung thực hoàn hảo)
- 0.5 = Một phần bịa đặt
- 0.0 = Hoàn toàn bịa đặt

TÀI LIỆU: {context}
CÂU HỎI: {question}
CÂU TRẢ LỜI: {answer}

CHỈ TRẢ VỀ MỘT CON SỐ (ví dụ: 0.8). Không giải thích.""")
    
    chain = prompt | llm
    result = chain.invoke({"context": context[:1500], "question": question, "answer": answer})
    try:
        return float(''.join(c for c in result.content.strip() if c.isdigit() or c == '.') or '0')
    except:
        return 0.0

def judge_relevancy(question: str, answer: str, llm) -> float:
    """
    TIÊU CHÍ 2: ĐÚNG TRỌNG TÂM (Answer Relevancy)
    Câu trả lời có đúng trọng tâm câu hỏi không?
    """
    prompt = ChatPromptTemplate.from_template("""
Bạn là Giám khảo AI. Hãy chấm điểm xem CÂU TRẢ LỜI có trả lời đúng trọng tâm CÂU HỎI không.
Chấm điểm từ 0.0 đến 1.0:
- 1.0 = Trả lời chính xác câu hỏi
- 0.5 = Trả lời lan man, chỉ đúng 1 phần
- 0.0 = Hoàn toàn lạc đề

CÂU HỎI: {question}
CÂU TRẢ LỜI: {answer}

CHỈ TRẢ VỀ MỘT CON SỐ. Không giải thích.""")
    
    chain = prompt | llm
    result = chain.invoke({"question": question, "answer": answer})
    try:
        return float(''.join(c for c in result.content.strip() if c.isdigit() or c == '.') or '0')
    except:
        return 0.0

def judge_correctness(answer: str, ground_truth: str, llm) -> float:
    """
    TIÊU CHÍ 3: ĐÚNG ĐÁP ÁN (Correctness)
    Câu trả lời có khớp với đáp án chuẩn không?
    """
    prompt = ChatPromptTemplate.from_template("""
Bạn là Giám khảo AI. So sánh CÂU TRẢ LỜI với ĐÁP ÁN CHUẨN xem có khớp ý nhau không.
Chấm điểm từ 0.0 đến 1.0:
- 1.0 = Khớp hoàn toàn (cho dù diễn đạt khác nhau)
- 0.5 = Đúng một phần
- 0.0 = Sai hoàn toàn

ĐÁP ÁN CHUẨN: {ground_truth}
CÂU TRẢ LỜI: {answer}

CHỈ TRẢ VỀ MỘT CON SỐ. Không giải thích.""")
    
    chain = prompt | llm
    result = chain.invoke({"answer": answer, "ground_truth": ground_truth})
    try:
        return float(''.join(c for c in result.content.strip() if c.isdigit() or c == '.') or '0')
    except:
        return 0.0

# ============================================================
# HÀM CHẠY TOÀN BỘ BÀI THI
# ============================================================
def run_evaluation():
    print("="*70)
    print("🏫 BẮT ĐẦU KỲ THI ĐÁNH GIÁ CHẤT LƯỢNG RAG CHATBOT")
    print("="*70)
    
    llm_judge = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    all_scores = []
    
    for i, test in enumerate(TEST_CASES):
        print(f"\n{'─'*50}")
        print(f"📝 Câu {i+1}/{len(TEST_CASES)}: {test['question']}")
        print(f"{'─'*50}")
        
        # 1. Lôi tài liệu ra (Retrieval)
        retrieved_docs = hybrid_search(test["question"], top_k=3)
        context = "\n\n".join(retrieved_docs)
        
        # 2. Hỏi Bot trả lời
        answer = generate_answer(test["question"])
        print(f"🤖 Bot trả lời: {answer[:150]}...")
        
        # 3. Giám Khảo AI chấm 3 tiêu chí
        faith = judge_faithfulness(test["question"], answer, context, llm_judge)
        relev = judge_relevancy(test["question"], answer, llm_judge)
        correct = judge_correctness(answer, test["ground_truth"], llm_judge)
        
        avg = round((faith + relev + correct) / 3, 2)
        
        print(f"   📊 Trung thực (Faithfulness):  {faith}")
        print(f"   📊 Đúng trọng tâm (Relevancy): {relev}")
        print(f"   📊 Đúng đáp án (Correctness):   {correct}")
        print(f"   ⭐ ĐIỂM TRUNG BÌNH:             {avg}")
        
        all_scores.append({
            "question": test["question"],
            "faithfulness": faith,
            "relevancy": relev,
            "correctness": correct,
            "average": avg
        })
    
    # Tổng kết
    avg_faith = round(sum(s["faithfulness"] for s in all_scores) / len(all_scores), 2)
    avg_relev = round(sum(s["relevancy"] for s in all_scores) / len(all_scores), 2)
    avg_correct = round(sum(s["correctness"] for s in all_scores) / len(all_scores), 2)
    avg_total = round(sum(s["average"] for s in all_scores) / len(all_scores), 2)
    
    print("\n" + "="*70)
    print("🏆 BẢNG ĐIỂM TỔNG KẾT CUỐI KỲ")
    print("="*70)
    print(f"   Trung thực (Faithfulness):  {avg_faith} / 1.0")
    print(f"   Đúng trọng tâm (Relevancy): {avg_relev} / 1.0")
    print(f"   Đúng đáp án (Correctness):   {avg_correct} / 1.0")
    print(f"   ═══════════════════════════")
    print(f"   ⭐ TỔNG ĐIỂM TRUNG BÌNH:     {avg_total} / 1.0")
    print("="*70)
    
    if avg_total >= 0.8:
        print("🎉 XẾP LOẠI: XUẤT SẮC! Bot sẵn sàng dùng THẬT!")
    elif avg_total >= 0.6:
        print("👍 XẾP LOẠI: KHÁ! Cần tinh chỉnh thêm Prompt/Chunking.")
    else:
        print("⚠️ XẾP LOẠI: YẾU! Cần xem lại dữ liệu và cách chia chunks.")
    
    return all_scores

if __name__ == "__main__":
    run_evaluation()
