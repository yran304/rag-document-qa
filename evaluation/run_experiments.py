"""One controlled experiment: compare top-k retrieval choices on the same dataset."""
from __future__ import annotations

import pandas as pd

from evaluation.evaluate import evaluate


def main() -> None:
    summaries = []
    for top_k in (2, 3, 5):
        frame = evaluate(chunk_size=800, overlap=150, top_k=top_k)
        summaries.append({
            "top_k": top_k,
            "retrieval_accuracy": frame["retrieval_success"].mean(),
            "answer_keyword_accuracy": frame["answer_keyword_match"].mean(),
            "faithfulness_proxy": frame["faithfulness_proxy"].mean(),
            "mean_latency_ms": frame["latency_ms"].mean(),
        })
    summary = pd.DataFrame(summaries)
    summary.to_csv("evaluation/results/top_k_experiment.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

