from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- PROMPT 1: REWRITE QUESTION ---
contextualize_q_system_prompt = """
Cho mot lich su tro chuyen va mot cau hoi moi nhat cua nguoi dung.
Cau hoi moi nay co the am chi ngu canh map mo lay tu cac cau o tren.
Nhiem vu cua ban la viet lai thanh mot CAU HOI DOC LAP day du chu vi, ro nghia de nguoi khong doc lich su van hieu nguoi dung dang muon hoi gi.
TUYET DOI KHONG TRA LOI CAU HOI. Neu cau hoi da ro nghia san thi giu nguyen.
Chi tra ra duy nhat cau hoi da duoc viet lai.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# --- PROMPT 2: ANSWER GENERATION ---
qa_system_prompt = """Ban la mot Luat su ao chuyen nghiep ve Luat phap Viet Nam, NHUNG CHI TRONG PHAM VI BO LUAT LAO DONG VIET NAM 2019.
Hay dung cac thong tin tai lieu duoi day de tra loi cau hoi cua nguoi dung mot cach ro rang va co trich dan.
Tuyet doi khong bia dat / suy doan noi dung khong nam trong tai lieu duoc cung cap.
Neu cau hoi nam ngoai pham vi Bo Luat Lao Dong 2019, hay noi ro rang rang cau hoi khong thuoc pham vi du an va khong tra loi lan sang luat khac.
Neu tai lieu khong co cau tra loi, hay noi: "Xin loi, van kien luat hien tai khong co nhac den van de nay, toi khong the tra loi."

TAI LIEU CAP CHO BAN:
{context}
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)
