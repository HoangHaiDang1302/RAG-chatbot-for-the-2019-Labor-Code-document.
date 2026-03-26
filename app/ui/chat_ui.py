import streamlit as st
import os
import sys

# Khai báo đường dẫn hệ thống
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services.generator import generate_answer
from langchain_core.messages import HumanMessage, AIMessage

# --- GIAO DIỆN ---
st.set_page_config(page_title="Trợ Lý AI Trí Nhớ Vô Cực", page_icon="🧠", layout="centered")
st.title("🧠 Trợ Lý Pháp Lý CÓ TRÍ NHỚ (Bản Víp)")
st.markdown("Giờ đây Khứa Luật Sư Llama 3 này đã ghi nhớ mọi câu bạn gáy với nó trước đó! (Hãy thử hỏi dồn dập xem nó có ngu ra không 😂)")

# --- TRÍ NHỚ RAM ---
# Thùng chứa 1 cho Streamlit hiện chữ cái Web GUI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Thùng chứa 2 là Bộ Não cho hệ thống Llama 3 tiêu hoá (Nó chỉ hiểu dạng mảng Đối tượng)
if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

# Reset Web (Vẽ lại màn hình khi F5)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Cục Prompt người dùng đập chát
user_input = st.chat_input("Hỏi dồn nó đi (VD câu 2: Vậy lúc đó có tính tiền không?)")

if user_input:
    # Bơm chữ màn hình Khách
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Tư duy câu trả lời
    with st.chat_message("assistant"):
        with st.spinner("Đang tua lại cuộn băng trí nhớ nãy giờ của 2 đứa mình... ⏳"):
            
            # CẤP NÃO!!! Ném nguyên cái máng trí nhớ nãy giờ qua cho Hàm RAG Tiêu Hoá
            coca_cola = generate_answer(user_input, chat_history=st.session_state.langchain_history)
            
            st.markdown(coca_cola)
            
    # Lưu lại câu trả lời vào màn hình Web 
    st.session_state.messages.append({"role": "assistant", "content": coca_cola})
    
    # CẬP NHẬT TRÍ NHỚ CHO LANGCHAIN !!!
    # Cứ mỗi vòng người gõ 1 câu - Bot trả 1 chữ, thì nhét 2 cục bộ nhớ này vào cuộn băng
    st.session_state.langchain_history.extend(
        [
            HumanMessage(content=user_input),   # Cấp thẻ Người nói
            AIMessage(content=coca_cola),       # Cấp thẻ Máy nói
        ]
    )
