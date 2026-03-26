from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Hàm này chia danh sách các Document (từ Bước 1) thành các đoạn văn bản (chunk) nhỏ hơn.
    - chunk_size: Kích thước tối đa của mỗi đoạn (tính bằng số ký tự).
    - chunk_overlap: Số ký tự trùng lặp giữa 2 đoạn liền kề (để không bị mất ngữ cảnh giữa 2 câu).
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Ưu tiên cắt theo đoạn văn, rồi đến dòng, rồi đến dấu cách
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Đã chia nhỏ tài liệu thành tổng cộng {len(chunks)} khối văn bản nhỏ (chunks).")
    return chunks

# Đoạn code dưới này để test chạy thử Bước 2
if __name__ == "__main__":
    import os
    import sys
    
    # Cấu hình đường dẫn để có thể gọi file document_loader trong cùng thư mục
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.services.document_loader import load_document
    
    # Khai báo lại file luật
    sample_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Bo_luat_lao_dong_2019.docx")
    sample_file = os.path.abspath(sample_file)
    
    try:
        # Bước 1: Load lại tài liệu
        print("Đang chạy Bước 1: Đọc tài liệu...")
        documents = load_document(sample_file)
        
        # Bước 2: Cắt nhỏ tài liệu
        print("\nĐang chạy Bước 2: Băm nhỏ tài liệu...")
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)
        
        # In thử ra xem các mảnh vụn trông như thế nào
        if len(chunks) > 1:
            print("\n--- XEM THỬ CHUNK SỐ 5 (Ví dụ) ---")
            print(f"(Dài {len(chunks[5].page_content)} ký tự)\n")
            print(chunks[5].page_content)
            
            print("\n--- XEM THỬ CHUNK SỐ 6 (Khối tiếp theo để xem phần text giao nhau) ---")
            print(f"(Dài {len(chunks[6].page_content)} ký tự)\n")
            print(chunks[6].page_content)
        
    except Exception as e:
        print(f"Lỗi: {e}")
