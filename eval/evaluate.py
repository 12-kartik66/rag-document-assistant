"""Evaluate the RAG pipeline with ragas.

Usage:
    pip install -r eval/requirements.txt
    python -m eval.evaluate [--questions data/questions.json] [--output results.json]
"""

import argparse
import json
import sys
from pathlib import Path

from langchain_groq import ChatGroq

from app.config import EVAL_MODEL, GROQ_API_KEY
from app.generation.qa_chain import retrieve_docs, answer_from_docs
from app.retrieval.vectorstore import get_vectorstore
from app.retrieval.embedder import get_embeddings


def get_evaluator_llm():
    return ChatGroq(model=EVAL_MODEL, groq_api_key=GROQ_API_KEY)


def fetch_docs_for_question(vectorstore, question, top_k=None):
    return retrieve_docs(vectorstore, question, k=top_k)


def main():
    parser = argparse.ArgumentParser(description="Run ragas evaluation on the RAG pipeline")
    parser.add_argument("--questions", default=str(Path(__file__).parent / "test_questions.json"))
    parser.add_argument("--output", default=str(Path(__file__).parent / "results.json"))
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            Faithfulness,
            ResponseRelevancy,
            LLMContextPrecisionWithoutReference,
            LLMContextRecall,
        )
    except ImportError as e:
        sys.exit(f"Missing evaluation dependency: {e}. Run: pip install -r eval/requirements.txt")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    vs = get_vectorstore()

    records = []
    for item in questions:
        question = item["question"]
        docs = fetch_docs_for_question(vs, question)
        result = answer_from_docs(docs, question)
        records.append({
            "question": question,
            "answer": result["answer"],
            "contexts": [d.page_content for d in docs],
            "ground_truth": item["ground_truth"],
        })

    metrics = [
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
        LLMContextRecall(),
    ]
    eval_llm = LangchainLLMWrapper(
        get_evaluator_llm(), bypass_n=True, bypass_temperature=True
    )
    eval_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    result = evaluate(
        Dataset.from_list(records),
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        run_config=RunConfig(max_workers=1, timeout=180, max_retries=10, max_wait=30),
    )

    frame = result.to_pandas()
    scores = {}
    for metric in metrics:
        if metric.name not in frame.columns:
            continue
        values = frame[metric.name].dropna()
        name = "context_precision" if metric.name == "llm_context_precision_without_reference" else metric.name
        scores[name] = round(float(values.mean()), 4) if len(values) else None
    per_sample = [
        {key: (None if value != value else value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]
    output = {
        "scores": scores,
        "per_sample": per_sample,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps(output["scores"], indent=2))
    print(f"Full results written to {out_path}")


if __name__ == "__main__":
    main()
