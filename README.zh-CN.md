[English](README.md) | 中文

# Inspectable RAG Document Q&A Service

一个为 GenAI/RAG 面试准备的、刻意保持小而透明的 Python 项目。它不使用 LangChain：你可以直接看到文本如何被切块、如何生成 embedding、如何由 FAISS 检索，以及上下文如何进入 LLM prompt。

## 功能

`POST /documents` 接收文本并执行：清洗 → 分块 → embedding → FAISS 索引。

`POST /ask` 执行：问题 embedding → cosine similarity top-k 检索 → grounded prompt → OpenAI Responses API → 答案、来源与每段相似度分数。

此外有 11 条小型评测集、pandas CSV 输出、top-k 对照实验和单元测试。

## 快速开始

需要 Python 3.10+ 和一个 OpenAI API key。官方 SDK 会从 `OPENAI_API_KEY` 环境变量读取 key；本项目将 key 放入未提交的 `.env` 文件。参考 [official OpenAI quickstart](https://platform.openai.com/docs/quickstart/make-your-first-api-request)。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# 编辑 .env，填入 OPENAI_API_KEY
uvicorn app.main:app --reload
```

打开 `http://127.0.0.1:8000/docs` 可试用 Swagger UI。先上传文件：

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'Content-Type: application/json' \
  -d '{"filename":"handbook.txt","text":"Full-time employees receive 20 paid vacation days."}'
```

再提问：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many vacation days do employees receive?"}'
```

服务退出时会把 FAISS 索引写入 `data/`；该目录被忽略，方便你重新建立索引。

## 代码地图

| 文件 | 责任 |
| --- | --- |
| `app/ingestion.py` | 文本规范化与带 overlap 的分块 |
| `app/embeddings.py` | 直接调用 provider 的 embedding API |
| `app/retrieval.py` | 向量 L2 归一化与 FAISS `IndexFlatIP` 检索 |
| `app/generation.py` | ground prompt 和 RAG orchestration |
| `app/main.py` | FastAPI 请求校验、错误处理、日志 |
| `evaluation/` | 可复现实验与 CSV 指标 |

## 测试和评测

```bash
pytest
python -m evaluation.evaluate --chunk-size 800 --overlap 150 --top-k 3
python -m evaluation.run_experiments
```

评测 CSV 的指标都是故意简单且可解释的：

- `retrieval_success`：期望的源文件是否进入 top-k。
- `answer_keyword_match`：答案是否包含预先写好的关键字；这是 correctness 的粗略代理，不是人工评审的替代。
- `faithfulness_proxy`：这些关键字是否也存在于被检索到的上下文，衡量答案的最低限度可追溯性。
- `latency_ms`：从 query embedding 到生成完成的耗时。

运行实验后，在 `evaluation/results/top_k_experiment.csv` 比较 top-k=2/3/5：top-k 较大可能提升召回，但也会扩大 prompt、增加延迟和噪声。不要预设结果，运行后如实记录结果。

### 已完成的真实运行（2026-08-14）

使用仓库中的 11 条评测问题、`chunk_size=800`、`overlap=150`、`top_k=3`，真实 API 运行的结果为：source retrieval success **100%**、keyword answer match **100%**、faithfulness proxy **100%**、平均端到端 latency **2395 ms**。top-k 对照实验的结果已保存为 `evaluation/results/top_k_experiment.csv`：top-k=2/3/5 的质量指标在这套很短的 demo 文档上均为 100%，平均 latency 分别约为 2121/1410/1513 ms。结论是该数据集不足以显示 top-k 的质量差异，而不是“top-k 一定不会影响结果”。

## 关键设计决定

FAISS 本身只计算向量内积。这里先对文档向量和问题向量都做 L2 归一化，因此内积就等于 cosine similarity。较高分数表示语义方向更相近，`top_k` 决定送给 LLM 的候选段数。

Grounding prompt 明确限定“只能依据上下文”，并要求信息不足时按固定句子回答。这能降低幻觉，但不能保证完全消除：若检索错了、文档本身有误或模型不完全服从指令，答案仍可能不好，所以需要评测和查看 `retrieved_chunks`。

更完整的面试讲解在 [INTERVIEW_GUIDE.zh-CN.md](INTERVIEW_GUIDE.zh-CN.md)。
