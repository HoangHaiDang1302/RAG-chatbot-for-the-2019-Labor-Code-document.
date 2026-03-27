import os
import sys

import streamlit as st

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from app.services.generator import generate_answer
from langchain_core.messages import AIMessage, HumanMessage

st.set_page_config(page_title="Trợ Lý Pháp Lý AI", page_icon="⚖️", layout="centered")
st.title("⚖️ Trợ Lý Pháp Lý Có Trích Dẫn")
st.markdown("Tra cứu và giải đáp Bộ Luật Lao Động 2019. Mỗi câu trả lời sẽ đi kèm nguồn tham khảo.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "langchain_history" not in st.session_state:
    st.session_state.langchain_history = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Hỏi về Luật Lao Động 2019...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất tài liệu và tạo câu trả lời..."):
            answer = generate_answer(user_input, chat_history=st.session_state.langchain_history)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.langchain_history.extend(
        [
            HumanMessage(content=user_input),
            AIMessage(content=answer),
        ]
    )
