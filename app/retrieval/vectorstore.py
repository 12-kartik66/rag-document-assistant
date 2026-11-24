from langchain_core.documents import Document
from langchain_chroma import Chroma
from app.retrieval.embedder import get_embeddings
from app.config import CHROMA_PERSIST_DIR


_source_cache = {}
_doc_cache = {}
_head_cache = {}


def get_vectorstore(persist_dir: str = CHROMA_PERSIST_DIR):
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def add_chunks(vectorstore, chunks: list[dict]):
    texts = []
    metadatas = []
    seen = set()
    for chunk in chunks:
        text = chunk.get("content", "")
        if not text or not text.strip():
            continue
        metadata = dict(chunk.get("metadata") or {})
        key = (metadata.get("source"), metadata.get("page"), text[:200])
        if key in seen:
            continue
        seen.add(key)
        texts.append(text)
        metadatas.append(metadata)
    if texts:
        vectorstore.add_texts(texts=texts, metadatas=metadatas)


def get_all_sources(vectorstore, use_cache: bool = False) -> list[str]:
    key = id(vectorstore)
    if use_cache and key in _source_cache:
        return list(_source_cache[key])

    collection = vectorstore.get(include=["metadatas"])
    metadatas = collection.get("metadatas") or [] if isinstance(collection, dict) else []
    sources = set()
    for m in metadatas:
        if m and "source" in m:
            sources.add(m["source"])
    result = sorted(sources)

    if use_cache:
        _source_cache[key] = result
    return result


def get_all_documents(vectorstore, use_cache: bool = False) -> list:
    key = id(vectorstore)
    if use_cache and key in _doc_cache:
        return list(_doc_cache[key])

    collection = vectorstore.get(include=["documents", "metadatas"])
    documents = []
    if isinstance(collection, dict):
        for text, metadata in zip(
            collection.get("documents") or [], collection.get("metadatas") or []
        ):
            if isinstance(text, str):
                documents.append(Document(page_content=text, metadata=dict(metadata or {})))

    if use_cache:
        _doc_cache[key] = documents
    return documents


def get_document_heads(vectorstore, source: str = None) -> list:
    key = (id(vectorstore), source)
    if key in _head_cache:
        return list(_head_cache[key])

    heads = {}
    for doc in get_all_documents(vectorstore, use_cache=True):
        metadata = doc.metadata or {}
        name = metadata.get("source")
        if source is not None and name != source:
            continue
        current = heads.get(name)
        if current is None:
            heads[name] = doc
            continue
        current_page = current.metadata.get("page")
        page = metadata.get("page")
        if current_page is None and page is not None:
            heads[name] = doc
        elif current_page is not None and page is not None and page < current_page:
            heads[name] = doc

    result = list(heads.values())
    _head_cache[key] = result
    return result


def clear_source_cache(vectorstore=None):
    if vectorstore is None:
        _source_cache.clear()
        _doc_cache.clear()
        _head_cache.clear()
    else:
        key = id(vectorstore)
        _source_cache.pop(key, None)
        _doc_cache.pop(key, None)
        for cache_key in [k for k in _head_cache if k[0] == key]:
            _head_cache.pop(cache_key, None)
