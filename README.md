English | [中文](README.zh-CN.md)

# Inspectable RAG Document Q&A Service

A deliberately small and transparent Python project built for GenAI/RAG interview prep. It does not use LangChain: you can see exactly how text gets chunked, how embeddings are generated, how FAISS performs retrieval, and how context flows into the LLM prompt.

## Features

`POST /documents` accepts text and runs it through: clean → chunk → embed → FAISS index.

`POST /ask` runs: question embedding → cosine similarity top-k retrieval → grounded prompt → OpenAI Responses API → answer, sources, and per-chunk similarity scores.

There's also an 11-question evaluation set, pandas CSV output, a top-k comparison experiment, and unit tests.

## Quickstart

Requires Python 3.10+ and an OpenAI API key. The official SDK reads the key from the `OPENAI_API_KEY` environment variable; this project keeps it in an untracked `.env` file. See the [official OpenAI quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and set OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` to try the Swagger UI. First, upload a document:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"filename":"handbook.txt","text":"Full-time employees receive 20 paid vacation days."}'
```

Then ask a question:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many vacation days do employees receive?"}'
```

On shutdown, the service writes the FAISS index to `data/`; that directory is gitignored so you can rebuild the index freely.

## Code map

| File | Responsibility |
| --- | --- |
| `app/ingestion.py` | Text normalization and overlap-aware chunking |
| `app/embeddings.py` | Direct calls to the provider's embedding API |
| `app/retrieval.py` | Vector L2 normalization and FAISS `IndexFlatIP` retrieval |
| `app/generation.py` | Grounded prompt construction and RAG orchestration |
| `app/main.py` | FastAPI request validation, error handling, logging |
| `evaluation/` | Reproducible experiments and CSV metrics |

## Tests and evaluation

```bash
pytest
python -m evaluation.evaluate --chunk-size 800 --overlap 150 --top-k 3
python -m evaluation.run_experiments
```

The evaluation CSV metrics are intentionally simple and interpretable:

- `retrieval_success`: whether the expected source document made it into the top-k results.
- `answer_keyword_match`: whether the answer contains pre-written keywords; this is a rough proxy for correctness, not a replacement for human review.
- `faithfulness_proxy`: whether those same keywords also appear in the retrieved context, measuring the minimal traceability of the answer.
- `latency_ms`: time from query embedding to generation completion.

After running the experiments, compare top-k=2/3/5 in `evaluation/results/top_k_experiment.csv`: a larger top-k may improve recall but also grows the prompt, latency, and noise. Don't assume the outcome — run it and record what actually happens.

### Latest real run (2026-08-14)

Using the repo's 11 evaluation questions with `chunk_size=800`, `overlap=150`, `top_k=3`, a real API run produced: source retrieval success **100%**, keyword answer match **100%**, faithfulness proxy **100%**, average end-to-end latency **2395 ms**. The top-k comparison experiment is saved at `evaluation/results/top_k_experiment.csv`: quality metrics for top-k=2/3/5 were all 100% on this short demo document set, with average latencies of roughly 2121/1410/1513 ms respectively. The takeaway is that this dataset is too small to reveal quality differences from top-k — not that "top-k doesn't affect results."

## Key design decisions

FAISS itself only computes inner products. Both document and query vectors are L2-normalized here, so the inner product becomes mathematically equivalent to cosine similarity. A higher score means the vectors point in a more similar semantic direction; `top_k` determines how many candidate chunks go to the LLM.

The grounding prompt explicitly restricts the model to "answer only from the provided context" and requires a fixed response when information is insufficient. This reduces hallucination but doesn't guarantee it away: if retrieval picks the wrong chunks, the source document itself is wrong, or the model doesn't fully follow instructions, the answer can still be bad — which is why evaluation and inspecting `retrieved_chunks` matter.

A more complete interview walkthrough (in Chinese) is available in [INTERVIEW_GUIDE.zh-CN.md](INTERVIEW_GUIDE.zh-CN.md).
