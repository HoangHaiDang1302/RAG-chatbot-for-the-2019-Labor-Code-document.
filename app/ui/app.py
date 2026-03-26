import streamlit as st
import os
import sys

# Khai báo đường dẫn để import lấy ruột của Chatbot
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from app.services.generator import generate_answer

# Cài đặt giao diện của trang Web
st.set_page_config(page_title="Trợ Lý Pháp Lý AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Trợ Lý Pháp Lý Ảo (Bộ Luật LĐ 2019)")
st.markdown("Hỏi tôi bất cứ câu nào về quyền lợi người lao động! (Chạy bằng Siêu Tốc Groq)")

# Khởi tạo kho lưu trữ tin nhắn ảo (phiên Chat của người dùng)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Trưng bày lại các tin nhắn cũ mỗi khi có tin nhắn mới (vì Web cứ reload liên tục)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Hộp tin nhắn để gõ (Prompt nhập của Người dùng)
user_input = st.chat_input("Ví dụ: Thời gian làm thêm giờ vào ngày cuối tuần tính lương như thế nào?")

if user_input:
    # 1. Hiển thị tin nhắn người dùng lên màn hình Chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Để cho RAG bắt đầu đi bới móc luật và dùng LLM trả lời lại
    with st.chat_message("assistant"):
        # Spinner quay quay để hiện thi cho người dùng biết AI đang lục não
        with st.spinner("Đang lặn xuống Kho Lục Luật, rọi Model AI Groq đằng kia... ⏳"):
            # Gọi ngay cỗ máy RAG mà bạn cực khổ ghép từ nãy giờ ở Bước 5
            coca_cola = generate_answer(user_input)
            
            # Ghi lời khuyên của luật sư Trí Tuệ Nhân Tạo ra màn hình!
            st.markdown(coca_cola)
            
    # 3. Lưu câu trả lời vào bộ nhớ RAM trang web để không biến mất
    st.session_state.messages.append({"role": "assistant", "content": coca_cola})

