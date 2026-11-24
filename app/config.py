import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
EVAL_MODEL = os.getenv("EVAL_MODEL", "openai/gpt-oss-20b")
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "16"))
RETRIEVAL_FETCH_MULTIPLIER = int(os.getenv("RETRIEVAL_FETCH_MULTIPLIER", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
