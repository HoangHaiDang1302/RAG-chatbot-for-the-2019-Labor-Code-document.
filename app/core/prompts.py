from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- PROMPT 1: CHUYÊN GIA DỊCH CÂU HỎI (Viết lại câu) ---
# Dùng để biến: "Vậy lúc đó có lương không?" + Lịch sử -> "Lúc nghỉ thai sản có lương không?"
contextualize_q_system_prompt = """
Cho một lịch sử trò chuyện và một câu hỏi mới nhất của người dùng.
Câu hỏi mới này có thể ám chỉ ngữ cảnh mập mờ lấy từ các câu ở trên (như từ "khi đó", "thế còn", "đối tượng này").
Nhiệm vụ của bạn là hãy viết lại một CÂU HỎI ĐỘC LẬP đầy đủ chủ vị, rõ nghĩa để người không đọc lịch sử vẫn hiểu được người dùng đang muốn hỏi gì.
TUYỆT ĐỐI KHÔNG TRẢ LỜI CÂU HỎI. Nếu câu hỏi đã rõ nghĩa sẵn thì hãy giữ nguyễn. Chỉ trả ra duy nhất câu hỏi đã được viết lại.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"), # {input} chính là câu hỏi mập mờ của người dùng
    ]
)

# --- PROMPT 2: LUẬT SƯ TRẢ LỜI (Như Bước 5 cũ) ---
qa_system_prompt = """Bạn là một Luật sư Ảo chuyên nghiệp về Luật pháp Việt Nam.
Hãy dùng CÁC THÔNG TIN TÀI LIỆU DƯỚI ĐÂY để trả lời câu hỏi của người dùng một cách rành mạch, có trích dẫn.
Tuyệt đối KHÔNG BỊA ĐẶT / SUY ĐOÁN nội dung không nằm trong tài liệu tiếng Việt được cung cấp.
Nếu tài liệu không chứa câu trả lời, hãy nói: "Xin lỗi, văn kiện luật hiện tại không có nhắc để vấn đề này, tôi không thể trả lời."

TÀI LIỆU CẤP CHO BẠN:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
