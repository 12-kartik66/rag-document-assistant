import json
from pathlib import Path
from fastapi import APIRouter, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vectorstore import (
    get_vectorstore,
    add_chunks,
    get_all_sources,
    clear_source_cache,
)
from app.generation.qa_chain import answer_question, stream_answer, clear_retrieval_caches

router = APIRouter()

vectorstore = None


def get_vs():
    global vectorstore
    if vectorstore is None:
        vectorstore = get_vectorstore()
    return vectorstore


def normalize_k(k):
    if k is None:
        return None
    try:
        return max(1, min(int(k), 20))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="k must be an integer")


@router.post("/upload")
async def upload(file: UploadFile):
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / filename
    file_path.write_bytes(await file.read())

    try:
        pages = load_document(str(file_path))
    except ValueError as e:
        file_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))

    chunks = chunk_documents(pages)
    vs = get_vs()
    source = file_path.as_posix()
    for candidate in {source, str(Path(source))}:
        vs.delete(where={"source": candidate})
    add_chunks(vs, chunks)
    clear_source_cache(vs)
    clear_retrieval_caches()

    return {"status": "success", "filename": filename, "chunks_added": len(chunks)}


@router.get("/documents")
async def list_documents():
    vs = get_vs()
    return {"documents": get_all_sources(vs)}


@router.post("/ask")
async def ask(question: str, source: str = None, k: int = None):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    vs = get_vs()
    return answer_question(vs, question, source=source, k=normalize_k(k))


@router.post("/ask/stream")
async def ask_stream(question: str, source: str = None, k: int = None):
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    vs = get_vs()
    normalized_k = normalize_k(k)

    async def event_generator():
        try:
            sources = []
            answer_text = ""
            for content, chunk_sources in stream_answer(vs, question, source=source, k=normalized_k):
                if chunk_sources is not None:
                    sources = chunk_sources
                if content:
                    answer_text += content
                    yield "data: " + json.dumps({"type": "chunk", "content": content}) + "\n\n"
            if not answer_text:
                yield "data: " + json.dumps({"type": "answer", "content": "I don't have enough information in the provided documents.", "sources": sources}) + "\n\n"
            yield "data: " + json.dumps({"type": "sources", "content": sources}) + "\n\n"
            yield "data: " + json.dumps({"type": "done"}) + "\n\n"
        except Exception as e:
            yield "data: " + json.dumps({"type": "error", "content": str(e)}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
