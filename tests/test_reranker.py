from unittest.mock import MagicMock, patch
from app.retrieval.reranker import rerank


def test_rerank_reorders_by_score():
    doc1 = MagicMock()
    doc1.page_content = "irrelevant text"
    doc2 = MagicMock()
    doc2.page_content = "highly relevant text"

    with patch("app.retrieval.reranker.get_reranker") as mock_get:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = [0.1, 0.9]
        mock_get.return_value = mock_reranker
        result = rerank("test query", [doc1, doc2], top_n=2)

    assert result[0] == doc2
    assert result[1] == doc1


def test_rerank_limits_results():
    docs = [MagicMock(page_content=f"doc {i}") for i in range(10)]

    with patch("app.retrieval.reranker.get_reranker") as mock_get:
        mock_reranker = MagicMock()
        mock_reranker.predict.return_value = list(range(10))
        mock_get.return_value = mock_reranker
        result = rerank("query", docs, top_n=3)

    assert len(result) == 3


def test_rerank_empty_docs():
    result = rerank("query", [], top_n=4)
    assert result == []
