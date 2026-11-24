from app.ingestion.chunker import chunk_documents


def test_chunk_documents_basic():
    pages = [
        {"content": "This is a test document. " * 50, "metadata": {"source": "test.txt", "page": None}}
    ]
    chunks = chunk_documents(pages)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "content" in chunk
        assert "metadata" in chunk
        assert chunk["metadata"]["source"] == "test.txt"


def test_chunk_documents_preserves_metadata():
    pages = [
        {"content": "Page one content. " * 30, "metadata": {"source": "doc.pdf", "page": 1}},
        {"content": "Page two content. " * 30, "metadata": {"source": "doc.pdf", "page": 2}},
    ]
    chunks = chunk_documents(pages)
    sources = {c["metadata"]["page"] for c in chunks}
    assert 1 in sources
    assert 2 in sources


def test_chunk_documents_respects_size():
    long_text = "word " * 1000
    pages = [{"content": long_text, "metadata": {"source": "test.txt", "page": None}}]
    chunks = chunk_documents(pages, chunk_size=200, chunk_overlap=20)
    for chunk in chunks:
        assert len(chunk["content"]) <= 250


def test_chunk_documents_empty():
    pages = [{"content": "", "metadata": {"source": "empty.txt", "page": None}}]
    chunks = chunk_documents(pages)
    assert len(chunks) == 0
