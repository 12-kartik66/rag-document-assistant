import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.vectorstore import get_vectorstore, add_chunks, get_all_sources


def test_add_chunks():
    mock_vs = MagicMock()
    chunks = [
        {"content": "chunk one", "metadata": {"source": "test.txt", "page": 1}},
        {"content": "chunk two", "metadata": {"source": "test.txt", "page": 2}},
    ]
    add_chunks(mock_vs, chunks)
    mock_vs.add_texts.assert_called_once()
    call_kwargs = mock_vs.add_texts.call_args
    assert call_kwargs[1]["texts"] == ["chunk one", "chunk two"]
    assert call_kwargs[1]["metadatas"] == [
        {"source": "test.txt", "page": 1},
        {"source": "test.txt", "page": 2},
    ]


def test_get_all_sources():
    mock_vs = MagicMock()
    mock_vs.get.return_value = {
        "metadatas": [
            {"source": "a.pdf", "page": 1},
            {"source": "b.pdf", "page": 1},
            {"source": "a.pdf", "page": 2},
        ]
    }
    result = get_all_sources(mock_vs)
    assert result == ["a.pdf", "b.pdf"]

    mock_vs.get.return_value = {"metadatas": None}
    assert get_all_sources(mock_vs) == []
