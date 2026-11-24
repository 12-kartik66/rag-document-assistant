# RAG Document Assistant

A Retrieval-Augmented Generation (RAG) system that lets you upload documents (PDF/TXT/MD) and ask natural-language questions. Answers are grounded in your document content with source citations.

## Features

- **Multi-file ingestion** — upload PDF, TXT, or MD files and query across all of them
- **Cross-encoder re-ranking** — retrieved chunks are re-ranked with `ms-marco-MiniLM-L-6-v2` to improve answer relevance
- **Per-document filtering** — restrict a question to specific uploaded documents
- **Streaming responses** — answers stream token-by-token into the UI
- **RAGAS evaluation** — optional script to measure faithfulness, answer relevancy, context precision/recall
- **Docker deployment** — backend + frontend containerized with `docker compose`

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│   FastAPI     │────▶│  Ingestion   │
│   Frontend   │◀────│   Backend     │◀────│  Pipeline    │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐     ┌──────────────┐
                     │  ChromaDB    │◀────│  Embeddings  │
                     │  Vector DB   │     │  (MiniLM)    │
                     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐     ┌──────────────┐
                     │ Cross-Encoder │    │  Groq LLM    │
                     │ Re-ranker    │────▶│  (Llama 3.3) │
                     └──────────────┘     └──────────────┘
```

## Setup

### 1. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
copy .env.example .env
```
Edit `.env` and add your Groq API key.

### 4. Run the backend
```bash
uvicorn app.main:app --reload
```

### 5. Run the frontend
```bash
streamlit run frontend/streamlit_app.py
```

## Usage

1. Open the Streamlit UI at `http://localhost:8501`
2. Upload a PDF, TXT, or MD file via the sidebar
3. Click **Ingest** to process the document
4. Ask questions in the text input and get grounded answers with source citations

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/upload` | POST | Upload and ingest a document |
| `/ask` | POST | Ask a question, get answer + sources (`?source=` filters by document) |
| `/ask/stream` | POST | Ask a question, get an SSE streamed answer + sources |
| `/documents` | GET | List all ingested documents |
| `/health` | GET | Health check |

## Running with Docker

```bash
# from the project root
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:8501`

## Running Evaluation (RAGAS)

```bash
pip install -r eval/requirements.txt
python -m eval.evaluate
```

Scores are written to `eval/results.json`.

## Running Tests

```bash
pytest tests/ -v
```

## Tuning Answer Detail

- Use the **Detail level** slider in the sidebar of the UI (2–20 context chunks) per question
- Set the default with `RETRIEVAL_K` in `.env` (default `8`)
- Pass `?k=<number>` to `/ask` or `/ask/stream` per-request
