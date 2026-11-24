from pathlib import Path
from pypdf import PdfReader


def load_pdf(file_path: str) -> list[dict]:
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({
                "content": text,
                "metadata": {"source": Path(file_path).as_posix(), "page": i + 1}
            })
    return pages


def load_text(file_path: str) -> list[dict]:
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()
    return [{
        "content": content,
        "metadata": {"source": Path(file_path).as_posix(), "page": None}
    }]


def load_document(file_path: str) -> list[dict]:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(file_path)
    elif ext in (".txt", ".md", ".markdown"):
        return load_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
