import re
from functools import lru_cache
from pathlib import Path

from langchain_core.documents import Document
from langchain_groq import ChatGroq

from app.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    RETRIEVAL_K,
    RETRIEVAL_FETCH_MULTIPLIER,
    MAX_CONTEXT_CHARS,
    LLM_TEMPERATURE,
    LLM_TIMEOUT,
    LLM_MAX_RETRIES,
)
from app.retrieval.reranker import rerank
from app.retrieval.vectorstore import get_all_sources, get_all_documents, get_document_heads


INSUFFICIENT_ANSWER = "I don't have enough information in the provided documents."

PROMPT_TEMPLATE = """You are a helpful document-Q&A assistant. Answer the question using ONLY the context below.
If the answer isn't in the context, reply with exactly this sentence and nothing else:
"I don't have enough information in the provided documents."

Always cite the source immediately after each claim.

CITATION RULES (follow exactly):
- Use plain ASCII square brackets only: [filename p.N], or [filename] when there is no page.
- Copy the bracketed marker from the context verbatim.
- Never use 【】, parentheses, backticks, italics, or the word "source" in place of a marker.
- Never add spaces just inside the brackets. Correct: [notes.pdf p.2]. Wrong: [ notes.pdf p.2 ].
- Cite only the chunks you actually used. Never invent a filename or page.

Write a DETAILED, THOROUGH answer:
- Cover every relevant point found in the context
- Organize the answer into clear paragraphs or bullet points
- Do not invent facts that are not in the context

Context:
{context}

Question: {question}
"""


@lru_cache(maxsize=1)
def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=LLM_TEMPERATURE,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
    )


def format_location(metadata: dict) -> str:
    source = metadata.get("source") or "unknown"
    page = metadata.get("page")
    return f"{source} p.{page}" if page is not None else str(source)


def build_context(docs: list) -> str:
    return "\n\n".join(
        f"[{format_location(d.metadata)}]\n{d.page_content}" for d in docs
    )


def prepare_context(docs: list) -> tuple[str, list]:
    parts = []
    selected = []
    total = 0
    for doc in docs:
        text = f"[{format_location(doc.metadata)}]\n{doc.page_content}"
        if selected and total + len(text) > MAX_CONTEXT_CHARS:
            break
        parts.append(text)
        selected.append(doc)
        total += len(text)
    return "\n\n".join(parts), selected


_header_epoch = 0


@lru_cache(maxsize=8)
def _get_page_headers_cached(vectorstore, epoch):
    headers = {}
    for doc in get_all_documents(vectorstore, use_cache=True):
        metadata = doc.metadata or {}
        page = metadata.get("page")
        if page is None:
            continue
        key = (metadata.get("source"), page)
        if key in headers:
            continue
        first_line = next(
            (line.strip() for line in doc.page_content.splitlines() if line.strip()), ""
        )
        if first_line:
            headers[key] = first_line[:120]
    return headers


def get_page_headers(vectorstore) -> dict:
    return _get_page_headers_cached(vectorstore, _header_epoch)


def clear_retrieval_caches():
    global _header_epoch
    _header_epoch += 1
    _get_page_headers_cached.cache_clear()


def add_page_headers(vectorstore, docs: list) -> list:
    if not docs:
        return docs
    headers = get_page_headers(vectorstore)
    result = []
    for doc in docs:
        metadata = dict(doc.metadata or {})
        text = doc.page_content or ""
        header = headers.get((metadata.get("source"), metadata.get("page")))
        if header and header not in text[: len(header) + 8]:
            text = f"{header}\n{text}"
        result.append(Document(page_content=text, metadata=metadata))
    return result


def _unique_docs(docs: list) -> list:
    seen = set()
    result = []
    for doc in docs:
        metadata = doc.metadata or {}
        key = (metadata.get("source"), metadata.get("page"), (doc.page_content or "")[:200])
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


def _diversify(docs: list, k: int, per_page: int = 3) -> list:
    counts = {}
    selected = []
    for doc in docs:
        metadata = doc.metadata or {}
        page = metadata.get("page")
        key = (metadata.get("source"), page) if page is not None else (
            metadata.get("source"), doc.page_content[:80]
        )
        if counts.get(key, 0) >= per_page:
            continue
        selected.append(doc)
        counts[key] = counts.get(key, 0) + 1
        if len(selected) >= k:
            break
    return selected


_METADATA_HINTS = (
    "author",
    "title",
    "who wrote",
    "who published",
    "publication",
    "citation",
    "journal",
)


def _boost_document_heads(question: str, docs: list, resolved_source, vectorstore) -> list:
    lowered = question.lower()
    if not any(hint in lowered for hint in _METADATA_HINTS):
        return docs
    heads = get_document_heads(vectorstore, source=resolved_source)
    if resolved_source is None:
        doc_sources = {(d.metadata or {}).get("source") for d in docs}
        heads = [h for h in heads if (h.metadata or {}).get("source") in doc_sources]
    if not heads:
        return docs

    selected = [heads[0]]
    keys = {
        (d.metadata.get("source"), d.metadata.get("page"), d.page_content)
        for d in selected
    }
    for doc in docs:
        key = (doc.metadata.get("source"), doc.metadata.get("page"), doc.page_content)
        if key in keys:
            continue
        selected.append(doc)
        keys.add(key)
    return selected


def _normalize_chunk(text: str) -> str:
    text = (text or "").replace("【", "[").replace("】", "]")
    text = re.sub(r"\[\s+", "[", text)
    text = re.sub(r"\s+\]", "]", text)
    text = re.sub(r"(?i)\bSource\s*:\s*\[", "[", text)
    return text


def _repair_citations(answer: str, docs: list) -> str:
    if not answer:
        return answer

    text = _normalize_chunk(answer)
    exact_by_key = {}
    exact_by_name = {}
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or "")
        name = Path(source).name.lower()
        marker = f"[{format_location(metadata)}]"
        exact_by_name.setdefault(name, marker)
        exact_by_key[(name, metadata.get("page"))] = marker

    def replace(match):
        raw = match.group(1)
        parsed = _parse_markers(f"[{raw}]")
        if not parsed:
            return match.group(0)
        name, page = next(iter(parsed))
        return exact_by_key.get((name, page)) or exact_by_name.get(name) or f"[{raw.strip()}]"

    return re.sub(r"\[\s*([^\[\]\n]+?)\s*\]", replace, text)


def _parse_markers(answer: str) -> set:
    normalized = (answer or "").replace("【", "[").replace("】", "]")
    markers = set()
    for raw in re.findall(r"\[\s*([^\[\]\n]+?)\s*\]", normalized):
        text = re.sub(r"\s+", " ", raw).strip()
        match = re.search(r"\s+p\.?\s*(\d+)\s*$", text, re.IGNORECASE)
        page = int(match.group(1)) if match else None
        base = text[: match.start()] if match else text
        name = base.replace("\\", "/").rsplit("/", 1)[-1].strip().lower()
        if name:
            markers.add((name, page))
    return markers


def cited_sources(answer: str, docs: list) -> list:
    markers = _parse_markers(answer)
    cited = []
    for doc in docs:
        metadata = doc.metadata or {}
        source = str(metadata.get("source") or "")
        name = Path(source).name.lower()
        page = metadata.get("page")
        if (name, page) in markers:
            cited.append((source, page))
    if not cited:
        cited = [(d.metadata.get("source"), d.metadata.get("page")) for d in docs]
    return list(dict.fromkeys(cited))


def answer_from_docs(docs: list, question: str) -> dict:
    if not docs:
        return {"answer": INSUFFICIENT_ANSWER, "sources": []}

    context, used_docs = prepare_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    answer = _repair_citations(get_llm().invoke(prompt).content, used_docs)

    if "don't have enough information" in (answer or "").lower():
        return {"answer": answer, "sources": []}
    return {"answer": answer, "sources": cited_sources(answer, used_docs)}


def answer_question(vectorstore, question: str, k: int = None, use_reranker: bool = True, source: str = None):
    if k is None:
        k = RETRIEVAL_K
    docs = retrieve_docs(vectorstore, question, k=k, use_reranker=use_reranker, source=source)
    return answer_from_docs(docs, question)


def resolve_source(vectorstore, question: str, source: str = None) -> str | None:
    if source:
        return source
    lowered = question.lower()
    for candidate in get_all_sources(vectorstore, use_cache=True):
        path = Path(candidate)
        stem = path.stem.lower()
        distinctive_stem = any(ch.isdigit() or ch in " _-" for ch in stem)
        if path.name.lower() in lowered or (distinctive_stem and stem in lowered):
            return candidate
    return None


def retrieve_docs(vectorstore, question: str, k: int = None, use_reranker: bool = True, source: str = None) -> list:
    if k is None:
        k = RETRIEVAL_K
    fetch_k = k * RETRIEVAL_FETCH_MULTIPLIER if use_reranker else k
    kwargs = {}
    resolved_source = resolve_source(vectorstore, question, source)
    if resolved_source:
        kwargs["filter"] = {"source": resolved_source}
    docs = vectorstore.similarity_search(question, k=fetch_k, **kwargs)
    docs = _unique_docs(docs)

    if use_reranker and docs:
        docs = rerank(question, docs, top_n=k)

    docs = _boost_document_heads(question, docs, resolved_source, vectorstore)
    docs = _diversify(_unique_docs(docs), k)
    if docs:
        docs = add_page_headers(vectorstore, docs)
    return docs


def stream_answer(vectorstore, question: str, k: int = None, use_reranker: bool = True, source: str = None):
    """Yield (answer_chunk, sources) tuples; the last tuple carries the final sources."""
    docs = retrieve_docs(vectorstore, question, k=k, use_reranker=use_reranker, source=source)

    if not docs:
        yield "", []
        return

    context, used_docs = prepare_context(docs)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    answer_parts = []
    for chunk in get_llm().stream(prompt):
        content = _normalize_chunk(chunk.content)
        if not content:
            continue
        answer_parts.append(content)
        yield content, None

    answer = "".join(answer_parts)
    sources = [] if "don't have enough information" in answer.lower() else cited_sources(answer, used_docs)
    yield "", sources
