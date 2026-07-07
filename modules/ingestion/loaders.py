import os
import re

from langchain_community.document_loaders import Docx2txtLoader


def normalize_document_text(text: str) -> str:
    text = (text or "").replace("</break>", " ").replace("<break>", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


clean_docx_text = normalize_document_text
_clean_docx_text = normalize_document_text


def read_text_source(path: str) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as handle:
                return handle.read()
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.read()


read_plain_text_file = read_text_source
_read_plain_text_file = read_text_source


def load_document_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".docx":
        return Docx2txtLoader(path).load()[0].page_content
    if ext in {".txt", ".md"}:
        return read_text_source(path)
    raise ValueError(f"Unsupported source file type: {ext}")


load_source_text = load_document_text
_load_source_text = load_document_text

