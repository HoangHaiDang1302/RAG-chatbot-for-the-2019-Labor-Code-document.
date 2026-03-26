import unittest
import time
import os
import sys

# Dẫn đường vòng để import code của chúng ta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.generator import generate_answer
from langchain_core.messages import HumanMessage, AIMessage

class TestRAGChatbot(unittest.TestCase):
    
    def setUp(self):
        # Giả lập trước một ca tư vấn chuẩn bị sẵn (Mock Data)
        self.history = [
            HumanMessage(content="Chào luật sư, lao động nữ sinh đôi thì được nghỉ thai sản bao nhiêu tháng?"),
            AIMessage(content="Trường hợp sinh đôi, bạn được nghỉ 6 tháng thai sản tiêu chuẩn, cộng thêm 1 tháng cho mầm non thứ 2 trở đi. Tổng cộng 7 tháng.")
        ]
        print(f"\n[KHAI BÁO BỘ NHỚ]")
        print(f"Bệnh nhân: '{self.history[0].content}'")
        print(f"Luật sư:   '{self.history[1].content}'")

    def test_01_cau_hoi_doc_lap_latency(self):
        """
        Test Case 1: Lòi ra câu hỏi chẳng liên quan tới thai sản.
        Kỳ vọng: Cổng gác (Router) phải đánh Cờ 'NO', bay lướt khâu dịch, chạy siêu nhanh.
        """
        print("\n" + "="*50)
        print("▶️ TEST CASE 1: CÂU HỎI RẼ NHÁNH KHÔNG ÁM CHỈ")
        print("="*50)
        cau_hoi = "Cho em hỏi thêm là người làm việc ca đêm thì tính lương ra sao?"
        
        start_time = time.time()
        answer = generate_answer(query=cau_hoi, chat_history=self.history)
        latency = time.time() - start_time
        
        print(f"\n⏱️ TỔNG THỜI GIAN NHẢ CHỮ: {latency:.2f} giây")
        print(f"💬 TRẢ LỜI NGẮN: {answer[:150]}...")
        
        self.assertTrue(len(answer) > 20)
        self.assertNotIn("Sự cố", answer, "Mã có thể đang bị lỗi ngoại lệ (Exception)")

    def test_02_cau_hoi_map_mo_latency(self):
        """
        Test Case 2: Đánh đố bằng câu hỏi chắp và ám chỉ vào chữ 'bảy tháng'.
        Kỳ vọng: Cổng gác (Router) nhận diện được sự mập mờ -> Đánh Cờ 'YES'.
                 Mất thêm thời gian (thêm latency) để gọi sếp Llama dịch lại rõ nghĩa.
        """
        print("\n" + "="*50)
        print("▶️ TEST CASE 2: CÂU HỎI MẬP MỜ (ĐÒI HỎI NÃO BỘ DỊCH)")
        print("="*50)
        cau_hoi = "Thế trong 7 tháng đó người mẹ có được nhận đủ tiền công 100% không thế?"
        
        start_time = time.time()
        answer = generate_answer(query=cau_hoi, chat_history=self.history)
        latency = time.time() - start_time
        
        print(f"\n⏱️ TỔNG THỜI GIAN NHẢ CHỮ (SẼ LÂU HƠN DO PHẢI DỊCH): {latency:.2f} giây")
        print(f"💬 TRẢ LỜI NGẮN: {answer[:150]}...")
        
        self.assertTrue(len(answer) > 20)
        self.assertNotIn("Sự cố", answer, "Mã có thể đang bị lỗi ngoại lệ (Exception)")

if __name__ == '__main__':
    # Hạn chế test bừa bãi kẻo kẹt hệ thống
    unittest.main(verbosity=2)
