import os
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    for idx, chunk in enumerate(chunks, start=1):
        if chunk.metadata is None:
            chunk.metadata = {}
        chunk.metadata["chunk_index"] = idx

    print(f"✅ Đã chia nhỏ tài liệu thành tổng cộng {len(chunks)} khối văn bản nhỏ (chunks).")
    return chunks


if __name__ == "__main__":
    import sys

    sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
    from app.services.document_loader import load_document

    sample_file = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Bo_luat_lao_dong_2019.docx")
    sample_file = os.path.abspath(sample_file)

    try:
        print("Đang chạy Bước 1: Đọc tài liệu...")
        documents = load_document(sample_file)

        print("\nĐang chạy Bước 2: Băm nhỏ tài liệu...")
        chunks = split_documents(documents, chunk_size=1000, chunk_overlap=200)

        if len(chunks) > 1:
            print("\n--- Xem thử chunk số 5 (ví dụ) ---")
            print(f"(Dài {len(chunks[5].page_content)} ký tự)\n")
            print(chunks[5].page_content)

            print("\n--- Xem thử chunk số 6 (khối tiếp theo để xem phần text giao nhau) ---")
            print(f"(Dài {len(chunks[6].page_content)} ký tự)\n")
            print(chunks[6].page_content)

    except Exception as e:
        print(f"Lỗi: {e}")
