import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

def load_document(file_path: str):
    """
    Hàm này nhận đầu vào là đường dẫn của một file, 
    xác định loại file và sử dụng công cụ đọc phù hợp.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file tại đường dẫn: {file_path}")

    # Lấy đuôi file (ví dụ: .pdf, .txt, .docx)
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pdf':
        print("Đang đọc file PDF...")
        loader = PyPDFLoader(file_path)
    elif file_extension == '.txt':
        print("Đang đọc file TXT...")
        loader = TextLoader(file_path, encoding='utf-8')
    elif file_extension in ['.docx', '.doc']:
        print("Đang đọc file Word...")
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Chưa hỗ trợ định dạng file này: {file_extension}")

    # loader.load() sẽ trả về một danh sách các 'Document' 
    documents = loader.load()
    print(f"✅ Đã tải xong! Tìm thấy {len(documents)} trang/đoạn.")
    return documents

if __name__ == "__main__":
    # Đường dẫn trỏ tới Bộ luật lao động 2019 trong thư mục data/raw
    # Bạn đổi đuôi .pdf thành .docx hoặc .txt tuỳ theo file bạn tải về nhé
    sample_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Bo_luat_lao_dong_2019.docx")
    
    try:
        # Normalize path
        sample_file = os.path.abspath(sample_file)
        print(f"Đang tìm file tại: {sample_file}")
        
        docs = load_document(sample_file)
        
        # In thử ra 300 ký tự đầu tiên của trang đầu
        print("\n--- NỘI DUNG 300 KÝ TỰ ĐẦU TIÊN CỦA TRANG 1 ---")
        print(docs[0].page_content[:300])
        
    except Exception as e:
        print(f"Lỗi: {e}")
