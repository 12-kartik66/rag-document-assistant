import os
import json
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Document Assistant", layout="wide")
st.title("RAG Document Assistant")

st.sidebar.header("Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["pdf", "txt", "md"])

if uploaded_file and st.sidebar.button("Ingest"):
    with st.spinner("Processing document..."):
        res = requests.post(f"{API_URL}/upload", files={"file": uploaded_file}, timeout=600)
        if res.status_code == 200:
            data = res.json()
            st.sidebar.success(f"Added {data['chunks_added']} chunks from {data['filename']}")
        else:
            st.sidebar.error(f"Error: {res.text}")

st.sidebar.header("Documents")

documents = []
try:
    res = requests.get(f"{API_URL}/documents", timeout=10)
    if res.status_code == 200:
        documents = res.json().get("documents", [])
except requests.ConnectionError:
    pass

if documents:
    selected_sources = st.sidebar.multiselect(
        "Filter by document (empty = all)", options=documents
    )
else:
    st.sidebar.info("No documents ingested yet. Upload a file above.")
    selected_sources = []


def stream_response(question, source, k):
    params = {"question": question, "k": str(k)}
    if source:
        params["source"] = source
    with requests.post(f"{API_URL}/ask/stream", params=params, stream=True, timeout=600) as res:
        if res.status_code != 200:
            yield {"type": "error", "content": res.text}
            return
        data_lines = []
        for raw in res.iter_lines(decode_unicode=True):
            if raw == "":
                if data_lines:
                    payload = "".join(data_lines)
                    try:
                        yield json.loads(payload)
                    except json.JSONDecodeError:
                        pass
                    data_lines = []
            elif raw.startswith("data:"):
                data_lines.append(raw[5:].strip())


st.header("Ask a Question")
question = st.text_input("Enter your question about the uploaded documents")
detail = st.slider("Detail level (number of context chunks to retrieve)", min_value=2, max_value=20, value=8)

if question and st.button("Ask"):
    with st.spinner("Searching documents..."):
        answer_box = st.empty()
        answer_text = ""
        sources = []

        for event in stream_response(question, selected_sources[0] if selected_sources else None, detail):
            etype = event.get("type")
            if etype == "chunk":
                answer_text += event.get("content", "")
                answer_box.write(answer_text)
            elif etype == "sources":
                sources = event.get("content", [])
            elif etype == "answer":
                answer_text = event.get("content", "")
                answer_box.write(answer_text)
            elif etype == "error":
                st.error(event.get("content"))
                break

        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.write(f"- {src[0]} (page {src[1]})")