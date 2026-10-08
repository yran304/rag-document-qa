"""Run a small, transparent offline evaluation and save per-question results to CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd

from app.config import Settings
from app.embeddings import OpenAIEmbedder
from app.generation import OpenAIGenerator, RagService
from app.ingestion import load_text_documents
from app.retrieval import FaissVectorStore


def evaluate(chunk_size: int, overlap: int, top_k: int) -> pd.DataFrame:
    settings = Settings()
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required to run evaluation")
    chunks = load_text_documents(Path("documents"), chunk_size, overlap)
    service = RagService(
        OpenAIEmbedder(settings.openai_api_key, settings.embedding_model),
        OpenAIGenerator(settings.openai_api_key, settings.chat_model),
        FaissVectorStore(),
        top_k,
    )
    service.index(chunks)
    cases = json.loads(Path("evaluation/questions.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for case in cases:
        started = time.perf_counter()
        result = service.ask(case["question"], top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        answer = result.answer.lower()
        retrieved_text = " ".join(chunk.text.lower() for chunk in result.retrieved_chunks)
        keywords = [word.lower() for word in case["answer_keywords"]]
        rows.append({
            "question": case["question"],
            "expected_source": case["expected_source"],
            "retrieved_sources": ", ".join(result.sources),
            "retrieval_success": case["expected_source"] in result.sources,
            "answer_keyword_match": all(word in answer for word in keywords),
            # A deliberately simple proxy: answer keywords must appear in retrieved context.
            "faithfulness_proxy": all(word in retrieved_text for word in keywords),
            "latency_ms": round(elapsed_ms, 2),
            "answer": result.answer,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "top_k": top_k,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    results = evaluate(args.chunk_size, args.overlap, args.top_k)
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"results_chunk{args.chunk_size}_topk{args.top_k}.csv"
    results.to_csv(output, index=False)
    print(results[["retrieval_success", "answer_keyword_match", "faithfulness_proxy", "latency_ms"]].mean(numeric_only=True))
    print(f"Saved detailed results to {output}")


if __name__ == "__main__":
    main()

