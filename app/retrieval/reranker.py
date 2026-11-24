from functools import lru_cache

from sentence_transformers import CrossEncoder
from app.config import RERANKER_MODEL, RERANK_BATCH_SIZE


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, docs: list, top_n: int = 4) -> list:
    if not docs:
        return []

    reranker = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs, batch_size=RERANK_BATCH_SIZE)

    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, score in scored_docs[:top_n]]
