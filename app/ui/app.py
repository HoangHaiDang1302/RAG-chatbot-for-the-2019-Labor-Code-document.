import os
import sys

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services.generator import generate_answer

st.set_page_config(page_title="Trợ Lý Pháp Lý AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Trợ Lý Pháp Lý AI")
st.markdown("Hỏi đáp về Bộ Luật Lao Động 2019 với nguồn tham khảo đi kèm.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Hỏi về Luật Lao Động 2019...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm nguồn và tạo câu trả lời..."):
            answer = generate_answer(user_input)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
