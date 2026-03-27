import os
import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

ARTICLE_HINT_RE = re.compile(r"^(điều|khoản|chương|mục)\s+\d+", re.IGNORECASE)


def _first_meaningful_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _extract_section_hint(text: str) -> str:
    first_line = _first_meaningful_line(text)
    if not first_line:
        return ""
    if ARTICLE_HINT_RE.search(first_line):
        return first_line[:120]
    return first_line[:120]


def split_documents(documents, chunk_size: int = 800, chunk_overlap: int = 120):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\nĐiều ",
            "\nĐiều ",
            "\n\nKhoản ",
            "\nKhoản ",
            "\n\nChương ",
            "\nChương ",
            "\n\nMục ",
            "\nMục ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    for idx, chunk in enumerate(chunks, start=1):
        if chunk.metadata is None:
            chunk.metadata = {}

        source_path = chunk.metadata.get("source", "data/raw")
        source_name = os.path.basename(source_path) if source_path else "data/raw"
        page = chunk.metadata.get("page")

        chunk.metadata["chunk_index"] = idx
        chunk.metadata["source_name"] = source_name
        chunk.metadata["section_hint"] = _extract_section_hint(chunk.page_content)
        if page is not None:
            chunk.metadata["page_number"] = page + 1 if isinstance(page, int) else page

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
        chunks = split_documents(documents)

        if len(chunks) > 1:
            print("\n--- Xem thử chunk số 5 (ví dụ) ---")
            print(f"(Dài {len(chunks[5].page_content)} ký tự)\n")
            print(chunks[5].page_content)

            print("\n--- Xem thử chunk số 6 (khối tiếp theo để xem phần text giao nhau) ---")
            print(f"(Dài {len(chunks[6].page_content)} ký tự)\n")
            print(chunks[6].page_content)

    except Exception as e:
        print(f"Lỗi: {e}")
