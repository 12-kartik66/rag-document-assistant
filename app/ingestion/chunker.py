import re

from langchain_text_splitters import RecursiveCharacterTextSplitter


def _markdown_sections(text: str) -> list:
    sections = []
    heading = ""
    buffer = []
    for line in text.splitlines():
        if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
            if buffer:
                sections.append((heading, "\n".join(buffer)))
                buffer = []
            heading = line.strip()
        buffer.append(line)
    if buffer:
        sections.append((heading, "\n".join(buffer)))
    return sections


def chunk_documents(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = []
    for page in pages:
        content = page["content"]
        source = str(page["metadata"].get("source", "")).lower()
        if source.endswith((".md", ".markdown")):
            sections = _markdown_sections(content)
        else:
            sections = [("", content)]

        for heading, section in sections:
            prefix = f"{heading}\n" if heading else ""
            for chunk_text in splitter.split_text(section):
                text = chunk_text.strip()
                if not text:
                    continue
                metadata = dict(page["metadata"])
                if heading:
                    metadata["heading"] = heading.lstrip("#").strip()
                chunks.append({
                    "content": prefix + text,
                    "metadata": metadata
                })
    return chunks
