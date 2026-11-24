from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.generation.qa_chain import (
    PROMPT_TEMPLATE,
    answer_question,
    cited_sources,
    format_location,
    resolve_source,
)


def test_answer_question_with_docs():
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "The sky is blue."
    mock_doc1.metadata = {"source": "test.pdf", "page": 1}

    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Grass is green."
    mock_doc2.metadata = {"source": "test.pdf", "page": 2}

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [mock_doc1, mock_doc2]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The sky is blue and grass is green.")

    with patch("app.generation.qa_chain.rerank", side_effect=lambda q, docs, top_n: docs):
        with patch("app.generation.qa_chain.get_llm", return_value=mock_llm):
            result = answer_question(mock_vs, "What color is the sky?")

    assert "answer" in result
    assert "sources" in result
    assert result["answer"] == "The sky is blue and grass is green."
    assert len(result["sources"]) == 2
    mock_vs.similarity_search.assert_called_once_with("What color is the sky?", k=24)


def test_answer_question_no_reranker():
    mock_doc = MagicMock()
    mock_doc.page_content = "The sky is blue."
    mock_doc.metadata = {"source": "test.pdf", "page": 1}

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [mock_doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The sky is blue.")

    with patch("app.generation.qa_chain.get_llm", return_value=mock_llm):
        result = answer_question(mock_vs, "What color is the sky?", use_reranker=False)

    mock_vs.similarity_search.assert_called_once_with("What color is the sky?", k=8)


def test_answer_question_with_source_filter():
    mock_doc = MagicMock()
    mock_doc.page_content = "The sky is blue."
    mock_doc.metadata = {"source": "specific.pdf", "page": 1}

    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = [mock_doc]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="The sky is blue.")

    with patch("app.generation.qa_chain.rerank", side_effect=lambda q, docs, top_n: docs):
        with patch("app.generation.qa_chain.get_llm", return_value=mock_llm):
            result = answer_question(mock_vs, "What color is the sky?", source="specific.pdf")

    mock_vs.similarity_search.assert_called_once_with(
        "What color is the sky?", k=24, filter={"source": "specific.pdf"}
    )


def test_answer_question_no_docs():
    mock_vs = MagicMock()
    mock_vs.similarity_search.return_value = []

    result = answer_question(mock_vs, "random question")
    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []


def test_prompt_template_contains_context():
    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE


def test_format_location_with_and_without_page():
    assert format_location({"source": "a.pdf", "page": 2}) == "a.pdf p.2"
    assert format_location({"source": "a.txt", "page": None}) == "a.txt"


def test_cited_sources_only_returns_cited_documents():
    docs = [
        Document(page_content="", metadata={"source": "a.txt", "page": None}),
        Document(page_content="", metadata={"source": "b.pdf", "page": 13}),
        Document(page_content="", metadata={"source": "c.pdf", "page": 2}),
    ]
    answer = "Paris [a.txt] and the result on page 2 [c.pdf p.2]."
    assert cited_sources(answer, docs) == [("a.txt", None), ("c.pdf", 2)]


def test_cited_sources_falls_back_when_answer_has_no_markers():
    docs = [Document(page_content="", metadata={"source": "a.txt", "page": None})]
    assert cited_sources("Some answer", docs) == [("a.txt", None)]


def test_resolve_source_detects_filename_in_question():
    with patch("app.generation.qa_chain.get_all_sources", return_value=["data/raw/iot 6.pdf"]):
        assert resolve_source(MagicMock(), "What is in iot 6.pdf?") == "data/raw/iot 6.pdf"
        assert resolve_source(MagicMock(), "What is the title?") is None
