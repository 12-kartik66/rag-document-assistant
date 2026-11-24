# RAG Document Assistant — Technical Documentation

A Retrieval-Augmented Generation (RAG) system where users upload documents (PDF / TXT / MD)
and ask natural-language questions. Answers are grounded strictly in the uploaded documents
and include source/page citations. If the answer is not in the documents, the system says so
instead of hallucinating.

---

## 1. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Language | Python 3.11+ | Core runtime |
| LLM | Groq `openai/gpt-oss-120b` | Answer generation |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Local, CPU-friendly vector embeddings |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-scores retrieved chunks for relevance |
| Vector DB | ChromaDB | Local vector store, persists to disk |
| Backend | FastAPI + Uvicorn | REST API + SSE streaming |
| Frontend | Streamlit | Web UI for upload and Q&A |
| Doc parsing | `pypdf` | PDF text extraction |
| Config | `python-dotenv` | Environment variables |

---

## 2. Folder Structure

```
RAG Document Assistant/
├── app/                          # Backend application package
│   ├── __init__.py
│   ├── main.py                   # FastAPI entrypoint
│   ├── config.py                 # Env vars + constants
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # /upload, /ask, /ask/stream, /documents endpoints
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py             # Load PDF / TXT / MD into page dicts
│   │   └── chunker.py            # Split page text into overlapping chunks
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── embedder.py           # HuggingFace embedding model wrapper
│   │   ├── vectorstore.py        # Chroma setup + add/query helpers
│   │   └── reranker.py           # Cross-encoder re-ranking
│   └── generation/
│       ├── __init__.py
│       └── qa_chain.py           # Prompt + retrieval + LLM (answer & stream)
├── frontend/
│   ├── streamlit_app.py          # Streamlit UI
│   ├── Dockerfile                # Frontend container image
│   └── requirements.txt          # Frontend-only deps
├── eval/
│   ├── __init__.py
│   ├── evaluate.py               # RAGAS evaluation script
│   ├── requirements.txt          # Evaluation-only deps
│   └── test_questions.json       # Q&A ground-truth set
├── tests/
│   ├── test_chunker.py           # Chunking unit tests
│   ├── test_retrieval.py         # Vector store unit tests
│   ├── test_reranker.py          # Re-ranker unit tests
│   └── test_qa_chain.py          # QA chain unit tests
├── data/
│   ├── raw/                      # Uploaded source documents
│   └── chroma_db/                # Persisted Chroma vector store
├── .env                          # Local secrets/config (not committed)
├── .env.example                  # Template for .env
├── .dockerignore
├── .gitignore
├── docker-compose.yml            # Backend + frontend orchestration
├── Dockerfile                    # Backend container image
├── requirements.txt              # Python dependencies
├── rag-document-assistant-build-guide.md
└── README.md
```

---

## 3. End-to-End Pipeline (Loader → Generation)

The system has two phases: **Ingestion** (write path) and **Query** (read path).

### 3.1 Ingestion Pipeline (upload → vector store)

```
 Uploaded File (PDF/TXT/MD)
          │
          ▼
 ┌─────────────────────┐
 │ routes.upload()     │  saves file to data/raw/
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ loader.load_document│  extracts text + {source, page} metadata
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ chunker.chunk_      │  splits text (size=500, overlap=50),
 │ documents()         │  preserves metadata per chunk
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ embedder.get_       │  HuggingFace MiniLM embedding model
 │ embeddings()        │
 └─────────┬───────────┘
           ▼
 ┌─────────────────────┐
 │ vectorstore.        │  embeds + stores chunks in ChromaDB
 │ add_chunks()        │  (persisted to data/chroma_db/)
 └─────────────────────┘
```

**Step by step:**

1. **Upload** — `POST /upload` receives the file, writes it to `data/raw/<filename>`.
2. **Load** — `load_document()` picks the loader by extension:
   - `.pdf` → `load_pdf()` uses `pypdf`, one dict per page (page number preserved).
   - `.txt` / `.md` / `.markdown` → `load_text()` reads whole file as a single page.
   - Anything else → raises `ValueError` (returned as HTTP 400).
3. **Chunk** — `chunk_documents()` runs `RecursiveCharacterTextSplitter` over each page's text,
   producing 500-char chunks with 50-char overlap and copying the page metadata onto each chunk.
4. **Embed + Store** — `add_chunks()` pulls `content` and `metadata` lists and calls
   `Chroma.add_texts()`, which embeds via MiniLM and persists to disk.

### 3.2 Query Pipeline (question → answer)

```
 User Question
      │
      ▼
 ┌──────────────────────┐
 │ retrieve_docs()      │  similarity_search(fetch_k = k*3 by default)
 │                      │  optional filter={"source": <file>}
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ reranker.rerank()    │  cross-encoder scores (query, chunk) pairs,
 │                      │  keeps top_n = k most relevant
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ Build context string │  "[source p.page]\\n<chunk>" joined by blank lines
 └──────────┬───────────┘
            ▼
 ┌──────────────────────┐
 │ qa_chain — LLM       │  Groq ChatLlama answer or token stream
 │ invoke()/stream()    │  using grounded PROMPT_TEMPLATE
 └──────────┬───────────┘
            ▼
   Answer + Sources
```

**Step by step:**

1. **Retrieve** — `retrieve_docs()` embeds the question and calls
   `vectorstore.similarity_search()`. It fetches `k * 3` candidates when the re-ranker is
   enabled (to give the re-ranker a wider pool), or just `k` when disabled. Results are scoped
   to a document when `source` is passed or when the question names an ingested filename.
2. **Re-rank** — `rerank()` runs the cross-encoder over each `(query, chunk)` pair, sorts by
   score descending, and returns the top `k`. The re-ranker model is loaded once and cached.
3. **Expand + build context** — The first chunk of each retrieved page is added when missing,
   so page titles and author lines are not lost. Each chunk is formatted as
   `[<source> p.<page>]` (or `[<source>]` for text files without a page number) followed by its
   text, joined with blank lines.
4. **Generate** —
   - `answer_question()` calls `llm.invoke(prompt)` and returns `{answer, sources}`.
   - `stream_answer()` calls `llm.stream(prompt)` and yields `(chunk_text, sources)` tuples for
     token-by-token streaming.
   - The `PROMPT_TEMPLATE` forces grounding, requires source/page citations, and instructs the
     model to reply *"I don't have enough information in the provided documents."* when the
     answer is absent.
5. **Sources** — Only the de-duplicated `(source, page)` tuples actually cited in the answer are
   returned; if the answer contains no markers, all context sources are returned as a fallback.

### 3.3 Streaming Flow (UI)

`POST /ask/stream` returns Server-Sent Events (SSE). `routes.ask_stream()` wraps
`stream_answer()` in an async generator and emits:

| Event | Payload |
|---|---|
| `chunk` | partial answer token(s) |
| `sources` | list of `[source, page]` pairs |
| `done` | stream finished |
| `error` | exception message |

The Streamlit frontend reads these events line-by-line and re-renders the growing answer.

---

## 4. File-by-File Function Reference

### `app/config.py`
Centralizes configuration loaded from `.env` via `python-dotenv`.

| Name | Description |
|---|---|
| `GROQ_API_KEY` | Groq API key (default `""`) |
| `CHROMA_PERSIST_DIR` | Chroma storage dir (default `./data/chroma_db`) |
| `EMBEDDING_MODEL` | Embedding model id (default `sentence-transformers/all-MiniLM-L6-v2`) |
| `RERANKER_MODEL` | Cross-encoder id (default `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `LLM_MODEL` | Groq model (default `openai/gpt-oss-120b`) |
| `EVAL_MODEL` | Groq model used as the RAGAS judge (default `openai/gpt-oss-20b`) |
| `RETRIEVAL_K` | Default number of chunks retrieved (default `8`) |

### `app/main.py`
| Function | Working |
|---|---|
| module-level `app` | Creates the `FastAPI` instance, attaches the router from `api/routes.py`. |
| `health()` | `GET /health` — returns `{"status": "ok"}` for liveness checks. |

### `app/api/routes.py`
| Function | Working |
|---|---|
| `get_vs()` | Lazily initializes and caches the global `vectorstore` (singleton). |
| `upload(file)` | `POST /upload` — saves upload to `data/raw/`, loads → chunks → embeds → stores; returns filename and chunk count. Raises HTTP 400 on unsupported type. |
| `list_documents()` | `GET /documents` — returns sorted list of ingested source filenames. |
| `ask(question, source, k)` | `POST /ask` — validates question, runs `answer_question()`, returns `{answer, sources}`. |
| `ask_stream(question, source, k)` | `POST /ask/stream` — returns SSE `StreamingResponse`; emits chunk/sources/done/error events. |
| `event_generator()` (inner) | Async generator that serializes each streamed event as `data: <json>\n\n`. |

### `app/ingestion/loader.py`
| Function | Working |
|---|---|
| `load_pdf(file_path)` | Reads a PDF with `pypdf`; returns one dict per non-empty page: `{content, metadata:{source, page}}`. |
| `load_text(file_path)` | Reads TXT/MD as UTF-8 (ignoring bad chars); returns a single page dict with `page=None`. |
| `load_document(file_path)` | Dispatches by file extension (`.pdf` / `.txt` / `.md` / `.markdown`); raises `ValueError` for unsupported types. |

### `app/ingestion/chunker.py`
| Function | Working |
|---|---|
| `chunk_documents(pages, chunk_size=500, chunk_overlap=50)` | Splits each page's `content` with `RecursiveCharacterTextSplitter` (separators `\n\n`, `\n`, `. `, ` `), skips blank chunks, and copies the page metadata onto every chunk. Returns a flat list of `{content, metadata}`. |

### `app/retrieval/embedder.py`
| Function | Working |
|---|---|
| `get_embeddings()` | Returns a `HuggingFaceEmbeddings` instance for `EMBEDDING_MODEL`, running locally on CPU. |

### `app/retrieval/vectorstore.py`
| Function | Working |
|---|---|
| `get_vectorstore(persist_dir=CHROMA_PERSIST_DIR)` | Creates/opens a persisted `Chroma` collection wired to the embedding function. |
| `add_chunks(vectorstore, chunks)` | Extracts `content`/`metadata` lists and calls `add_texts()` to embed + store them. |
| `get_all_sources(vectorstore)` | Reads all stored metadata and returns the de-duplicated, sorted set of source filenames (handles `None`). |

### `app/retrieval/reranker.py`
| Function | Working |
|---|---|
| `get_reranker()` | Returns a `CrossEncoder` for `RERANKER_MODEL`. |
| `rerank(query, docs, top_n=4)` | Scores each `(query, doc.page_content)` pair, sorts descending by score, returns the top `top_n` docs (empty list for empty input). |

### `app/generation/qa_chain.py`
| Function | Working |
|---|---|
| `get_llm()` | Returns a cached Groq `ChatGroq` client for `LLM_MODEL`. |
| `format_location(metadata)` | Renders `"<source> p.<page>"`, or just `"<source>"` when the page is unknown. |
| `build_context(docs)` | Joins docs into the `[<location>]` + text context block. |
| `get_page_headers(vectorstore)` | Map of `(source, page)` to the page's first chunk. |
| `add_page_headers(vectorstore, docs)` | Appends each retrieved page's first chunk when it is not already present (deduplicated by content). |
| `cited_sources(answer, docs)` | Returns the de-duplicated `(source, page)` pairs cited in the answer, falling back to all context sources. |
| `resolve_source(vectorstore, question, source)` | Returns `source` when given, otherwise the ingested filename named in the question, else `None`. |
| `retrieve_docs(vectorstore, question, k, use_reranker, source)` | Fetches `k*3` candidates (or `k`), scopes to a source when detected, re-ranks down to `k`, then adds page headers. |
| `answer_question(vectorstore, question, k, use_reranker, source)` | Retrieves docs, builds the context, formats the prompt, calls the LLM, and returns `{answer, sources}`. Returns the fallback message when no docs are found. |
| `answer_from_docs(docs, question)` | Same as above but skips retrieval, so evaluation can reuse the exact retrieved contexts. |
| `stream_answer(vectorstore, question, k, use_reranker, source)` | Same retrieval/prompting, but yields `(token, sources)` from `llm.stream()`; the final tuple carries the cited sources; yields `("", [])` if no docs. |
| `PROMPT_TEMPLATE` | Grounded-answer prompt instructing the model to copy the exact `[source p.page]` marker, be detailed, avoid invention, and admit when information is missing. |

### `frontend/streamlit_app.py`
| Function / Block | Working |
|---|---|
| `API_URL` | Backend base URL from env (default `http://localhost:8000`). |
| Upload widget | Sidebar file uploader + **Ingest** button posts to `/upload`. |
| Documents sidebar | `GET /documents` populates a multiselect for per-document filtering. |
| `stream_response(question, source, k)` | Posts to `/ask/stream` and parses SSE `data:` lines into JSON events. |
| Ask section | Question input + **Detail level** slider (2–20 chunks); streams the answer into an `st.empty()` box and lists sources in an expander. |

### `eval/evaluate.py`
| Function | Working |
|---|---|
| `fetch_docs_for_question(vectorstore, question, top_k=4)` | Retrieves contexts used as RAGAS `contexts`. |
| `get_evaluator_llm()` | Returns the Groq LLM wrapped for RAGAS scoring. |
| `main()` | Parses args, loads `test_questions.json`, runs the pipeline per question, builds a `datasets.Dataset`, computes `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, and writes `results.json`. |

---

## 5. API Endpoints

| Endpoint | Method | Params | Description |
|---|---|---|---|
| `/upload` | POST | `file` (multipart) | Ingest a PDF/TXT/MD document. Returns `{status, filename, chunks_added}`. |
| `/documents` | GET | — | List all ingested document sources. |
| `/ask` | POST | `question`, `source?`, `k?` | Return `{answer, sources}`. |
| `/ask/stream` | POST | `question`, `source?`, `k?` | SSE stream of `chunk`/`sources`/`done`/`error` events. |
| `/health` | GET | — | Liveness check. |

---

## 6. Configuration (`.env`)

```
GROQ_API_KEY=your_key_here
CHROMA_PERSIST_DIR=./data/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
LLM_MODEL=openai/gpt-oss-120b
RETRIEVAL_K=8
```

---

## 7. Running the Project

```bash
# Backend
uvicorn app.main:app --reload

# Frontend
streamlit run frontend/streamlit_app.py

# Tests
pytest tests/ -v

# Evaluation
pip install -r eval/requirements.txt
python -m eval.evaluate

# Docker (both services)
docker compose up --build
```

Backend: `http://localhost:8000` · Frontend: `http://localhost:8501`

---

## 8. Design Notes

- **Grounding over guessing** — the prompt and the no-docs fallback guarantee the model only
  answers from retrieved context.
- **Two-stage retrieval** — vector similarity gives breadth (`k*3`), the cross-encoder gives
  precision (top `k`), improving answer relevance over pure vector search.
- **Metadata-preserving chunks** — every chunk carries `{source, page}`, enabling precise
  citations and per-document filtering.
- **Lazy singleton vector store** — `get_vs()` avoids re-opening Chroma on every request.
- **Streaming first** — the primary UI path uses SSE for responsive token-by-token output.
