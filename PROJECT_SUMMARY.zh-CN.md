[English](PROJECT_SUMMARY.md) | 中文

# Inspectable RAG Document Q&A Service — 项目总结（简历/面试准备）

> 本文档基于当前代码库逐文件核查后生成，所有数字和文件引用均可在仓库中复核。第 8 节明确列出未实现/半成品部分，未做任何美化。

---

## 1. 一句话说明

一个不依赖 LangChain、刻意保持"可检查"（inspectable）的 RAG（Retrieval-Augmented Generation）文档问答服务：把纯文本文档切块、生成 embedding、用 FAISS 做向量检索，再把检索到的上下文交给 OpenAI 大模型生成有来源引用的答案。目标用户是**准备 GenAI/RAG 面试的开发者自己**——项目定位是学习/面试作品集，不是生产系统（README 原话："为 GenAI/RAG 面试准备的、刻意保持小而透明的 Python 项目"，见 [README.md:2-3](README.md)）。

## 2. 技术栈和版本（均取自 [requirements.txt](requirements.txt)）

| 类别 | 库 | 版本约束 |
| --- | --- | --- |
| Web 框架 | fastapi | `>=0.115,<1.0` |
| ASGI 服务器 | uvicorn[standard] | `>=0.30,<1.0` |
| LLM/Embedding SDK | openai | `>=1.0,<3.0` |
| 向量检索 | faiss-cpu | `>=1.8` |
| 数值计算 | numpy | `>=1.26,<3.0` |
| 数据分析/评测 | pandas | `>=2.2,<3.0` |
| 数据校验 | pydantic | `>=2.0,<3.0` |
| 环境变量加载 | python-dotenv | `>=1.0,<2.0` |
| 测试框架 | pytest | `>=8.0,<9.0` |
| HTTP 客户端（测试用） | httpx | `>=0.27,<1.0` |

语言：Python 3.10+（README 明确要求，代码使用 `from __future__ import annotations` 和 `X | None` 联合类型语法）。

外部服务：OpenAI API —— chat 模型默认 `gpt-4.1-mini`，embedding 模型默认 `text-embedding-3-small`（均可通过环境变量覆盖，见 [app/config.py:15-16](app/config.py:15)）。**没有使用 LangChain / LlamaIndex 等 RAG 框架**——这是项目刻意的设计选择，用于面试时能逐行讲清楚每一步在做什么。

## 3. 架构、目录结构与请求全流程

### 目录结构与模块职责

```
app/
  main.py        FastAPI 应用入口：路由、生命周期管理、错误处理、日志
  config.py      从环境变量读取配置（API key、模型名、chunk 参数、top_k）
  models.py      Pydantic 数据模型（请求/响应/领域对象）
  ingestion.py   文本清洗 + 带 overlap 的字符级分块
  embeddings.py  Embedding provider 封装（Protocol + OpenAI 实现）
  generation.py  RAG 编排（RagService）+ grounding prompt 构建 + LLM 调用封装
  retrieval.py   FAISS 向量存储：归一化、索引、检索、持久化
evaluation/
  evaluate.py         单次评测：跑 11 条问题，输出 CSV 指标
  run_experiments.py  对照实验：在 top_k=2/3/5 上重复评测并汇总
  questions.json      11 条人工标注的评测问题（含期望来源、期望关键词）
tests/           pytest 单元测试（分块、检索、服务编排、API）
documents/       两份示例纯文本文档（员工手册、产品发布说明）
data/            运行时生成的 FAISS 索引 + chunk 元数据（被 .gitignore 忽略）
```

### 典型请求的完整链路

**写入路径 `POST /documents`**（[app/main.py:48-58](app/main.py:48)）：
1. FastAPI 用 `DocumentInput` 校验 `filename`/`text`（[app/models.py:13-15](app/models.py:13)）。
2. `chunk_text()` 先用正则 `\s+` → 单空格做 `normalize_text`（[app/ingestion.py:10-12](app/ingestion.py:10)），再按 `chunk_size`/`chunk_overlap` 做字符级滑窗切分，并优先在空格处断句、避免从单词中间切开（[app/ingestion.py:15-48](app/ingestion.py:15)）。
3. `RagService.index()` 调用 `OpenAIEmbedder.embed()` 批量获取 embedding（[app/generation.py:50-54](app/generation.py:50), [app/embeddings.py:19-32](app/embeddings.py:19)）。
4. `FaissVectorStore.add()` 对向量做 L2 归一化后写入 `IndexFlatIP` 索引，并把 chunk 元数据追加到内存列表（[app/retrieval.py:31-42](app/retrieval.py:31)）。
5. 服务进程退出时（FastAPI `lifespan` 的收尾部分），索引和元数据被落盘到 `data/index.faiss` / `data/chunks.json`（[app/main.py:36](app/main.py:36)）。

**问答路径 `POST /ask`**（[app/main.py:61-70](app/main.py:61)）：
1. `AskRequest` 校验问题长度（1–2000 字符）和可选的 `top_k`（1–10）（[app/models.py:18-20](app/models.py:18)）。
2. `RagService.ask()` 把问题也 embed 成向量（[app/generation.py:65](app/generation.py:65)）。
3. `FaissVectorStore.search()` 对 query 向量做同样的 L2 归一化，再用 FAISS 做内积检索，内积在归一化后数学等价于 cosine similarity，返回 top-k 个 `RetrievedChunk`（含相似度分数）（[app/retrieval.py:44-53](app/retrieval.py:44)）。
4. `build_prompt()` 把 grounding instructions + 每个检索到的 chunk（标明 SOURCE/CHUNK/TEXT）+ 用户问题拼成一个 prompt 字符串（[app/generation.py:37-39](app/generation.py:37)）。
5. `OpenAIGenerator.generate()` 调用 OpenAI Responses API 生成答案（[app/generation.py:31-33](app/generation.py:31)）。
6. 若索引为空或检索无结果，直接返回固定的"信息不足"话术，不调用 LLM（[app/generation.py:61-67](app/generation.py:61)）。
7. 返回值包含答案、去重排序后的来源文件名列表、完整的检索 chunk（供调试/可解释性）、以及从 query embedding 开始到生成结束的端到端延迟（`time.perf_counter()` 差值，单位 ms）（[app/generation.py:56-77](app/generation.py:56)）。

## 4. 关键设计决策及原因

- **向量归一化 + 内积 = cosine similarity**：FAISS 的 `IndexFlatIP` 本身只算内积；项目在写入和查询两端都用 `faiss.normalize_L2` 做 L2 归一化，使内积数值上等价于 cosine similarity，从而用最简单的索引类型获得"方向相似度"的语义（[app/retrieval.py:1, 13-18](app/retrieval.py:1)）。这是一个刻意的简化：数据量大时应换用近似最近邻索引（如 `IndexIVFFlat`/HNSW），但对面试级别的小数据集，`IndexFlatIP`（精确暴力搜索）足够且延迟可控。
- **Grounding prompt 强约束幻觉，但不保证消除**：prompt 明确要求"只能依据上下文回答"，信息不足时输出固定句子（[app/generation.py:15-19](app/generation.py:15)）。README 明确指出这只是降低幻觉的手段，不是保证——如果检索错了、文档本身有误、或模型不完全服从指令，答案仍可能有问题，因此设计上把 `retrieved_chunks` 和相似度分数一并返回给调用方，方便审查（[README.md:78-80](README.md:78)）。
- **Provider 边界用 `Protocol` 抽象**：`Embedder` 和 `Generator` 都定义为 `typing.Protocol`（[app/embeddings.py:10-11](app/embeddings.py:10), [app/generation.py:22-23](app/generation.py:22)），`RagService` 只依赖这两个接口而不直接依赖 OpenAI SDK。这让 `tests/test_service.py` 里可以用 `FakeEmbedder`/`FakeGenerator` 完全离线测试编排逻辑，不需要真实 API key 或网络调用（[tests/test_service.py:8-17](tests/test_service.py:8)）。
- **一致性/并发处理是被有意简化的**：向量索引和 chunk 元数据只存在于**单进程内存**里（`FaissVectorStore` 是普通 Python 对象，[app/retrieval.py:21-29](app/retrieval.py:21)），仅在 FastAPI `lifespan` 启动时 `load()`、退出时 `save()` 做一次性持久化（[app/main.py:26-27, 36](app/main.py:26)）。这意味着：没有事务、没有并发写锁、多进程/多实例部署下各自维护独立索引且互不同步、进程崩溃会丢失自上次正常退出以来的所有写入。这是一个明确的取舍——用换取代码可读性和面试可讲解性，而非工程上的数据一致性保证（详见第 8、9 节）。
- **容错策略是"快速失败 + 转译为 HTTP 错误"**，不是重试/降级：`/documents` 和 `/ask` 都用 `try/except Exception` 捕获底层 SDK 异常，记录日志后转成 `502`（provider 失败）或 `422`（业务校验失败，如空问题）（[app/main.py:53-58, 64-70](app/main.py:53)）。没有超时控制、没有重试（retry/backoff）、没有熔断，调用方需要自己处理 OpenAI API 的限流或超时。
- **分块算法优先保证可读性而非语义完整性**：按固定字符数切分并在空格处回退断句，同时用 overlap 保留跨边界的一点上下文（[app/ingestion.py:26-47](app/ingestion.py:26)），不做基于句子/段落的语义分块，简单但对长句、代码块等结构不敏感。

## 5. 测试

**单元测试**（`pytest`，4 个测试文件，全部可离线运行，不需要 OpenAI API key，见 [tests/](tests/)）：

| 文件 | 覆盖场景 |
| --- | --- |
| [tests/test_chunking.py](tests/test_chunking.py) | 文本规范化（空白折叠）；分块保留来源文件名与索引；相邻 chunk 之间确实存在 overlap 内容 |
| [tests/test_retrieval.py](tests/test_retrieval.py) | `FaissVectorStore` 对未归一化的原始向量也能正确检索出语义最近的一条，分数 > 0.9 |
| [tests/test_service.py](tests/test_service.py) | 用 `FakeEmbedder`/`FakeGenerator` 验证 `RagService` 端到端编排：索引 → 检索 → 生成 prompt 包含 `DOCUMENT CONTEXT` → 返回正确来源 |
| [tests/test_api.py](tests/test_api.py) | 用 FastAPI `TestClient` 验证空索引时 `/ask` 返回 200 且给出"信息不足"的安全兜底答案 |

运行方式：
```bash
pytest
```

**离线评测**（不是单元测试，是针对真实 OpenAI API 的效果评测，需要 API key）：
```bash
python -m evaluation.evaluate --chunk-size 800 --overlap 150 --top-k 3
python -m evaluation.run_experiments
```
评测集是 [evaluation/questions.json](evaluation/questions.json) 里手工标注的 **11 条问题**，覆盖两份示例文档（员工手册 7 题、产品发布说明 4 题）。四个指标（[evaluation/evaluate.py:39-52](evaluation/evaluate.py:39)）：
- `retrieval_success`：期望来源文件是否出现在 top-k 检索结果里。
- `answer_keyword_match`：答案是否包含预设关键词——**是 correctness 的粗略代理，不是人工评审的替代**（README 原话，[README.md:66](README.md:66)）。
- `faithfulness_proxy`：这些关键词是否也出现在被检索到的上下文里，衡量答案的最低限度可追溯性，不是严格的 faithfulness 度量。
- `latency_ms`：从 query embedding 到生成完成的端到端耗时。

已记录的一次真实运行结果（2026-08-14，`chunk_size=800, overlap=150, top_k=3`）：source retrieval success **100%**、keyword answer match **100%**、faithfulness proxy **100%**、平均端到端延迟 **2395 ms**（[README.md:74](README.md:74)，CSV 见 [evaluation/results/results_chunk800_topk3.csv](evaluation/results/results_chunk800_topk3.csv)）。top-k 对照实验（top_k=2/3/5）在这套 11 题小数据集上质量指标均为 100%，平均延迟分别约 2121/1410/1513 ms（[evaluation/results/top_k_experiment.csv](evaluation/results/top_k_experiment.csv)）——**结论应表述为"这套数据集太小，不足以体现 top-k 对质量的影响"，而不是"top-k 不影响质量"**，这一点 README 本身也明确指出（[README.md:74](README.md:74)）。

## 6. 部署和运行方式

- **本地运行**：`venv` + `pip install -r requirements.txt` + `uvicorn app.main:app --reload`，通过 `.env` 文件（未提交到版本库）注入 `OPENAI_API_KEY`（[README.md:13-24](README.md:13), [app/config.py:7-9](app/config.py:7)）。
- **API 文档**：FastAPI 自带的 Swagger UI，`http://127.0.0.1:8000/docs`。
- **配置项**（均为环境变量，[app/config.py:14-19](app/config.py:14)）：`OPENAI_API_KEY`、`OPENAI_CHAT_MODEL`（默认 `gpt-4.1-mini`）、`OPENAI_EMBEDDING_MODEL`（默认 `text-embedding-3-small`）、`CHUNK_SIZE`（默认 800）、`CHUNK_OVERLAP`（默认 150）、`TOP_K`（默认 3）。
- **持久化**：无数据库，FAISS 索引和 chunk 元数据以文件形式存在 `data/` 目录，该目录被 `.gitignore` 排除。
- **Docker / CI/CD：没有**。仓库中没有 `Dockerfile`、`docker-compose.yml`，也没有任何 `.yml`/`.yaml` CI 配置文件（如 GitHub Actions）——这是当前明确的空白，不是被遗漏的检查。

## 7. 简历中可以准确写出的事实（附证据位置）

以下每一条都能在代码里直接指认，适合写进简历或在面试中被追问时展开：

- 用 **FastAPI + Pydantic** 实现了两个核心端点（`POST /documents` 文档摄取、`POST /ask` 检索问答）和一个健康检查端点，请求/响应均有类型校验（[app/main.py](app/main.py), [app/models.py](app/models.py)）。
- 手写实现了**带 overlap 的字符级文本分块算法**（非调库），并优先在词边界断句以保证可读性（[app/ingestion.py:15-48](app/ingestion.py:15)）。
- 使用 **FAISS `IndexFlatIP`** 构建向量索引，通过对 embedding 做 **L2 归一化**，用内积等价实现 **cosine similarity** 检索（[app/retrieval.py:13-18, 44-53](app/retrieval.py:13)）。
- 设计了基于 `Protocol` 的 **provider 抽象层**（`Embedder`/`Generator`），使核心业务逻辑 `RagService` 可在不依赖真实 OpenAI API 的情况下被单元测试完全覆盖（[app/embeddings.py:10-11](app/embeddings.py:10), [app/generation.py:22-23](app/generation.py:22), [tests/test_service.py](tests/test_service.py)）。
- 实现了 **grounding prompt**，限定模型仅依据检索到的上下文回答，并对信息不足的情况输出固定安全话术，降低幻觉风险（[app/generation.py:15-19](app/generation.py:15)）。
- 搭建了一套**可复现的离线评测流水线**：11 条人工标注问题、4 个可解释指标（检索命中率、关键词匹配率、faithfulness 代理指标、延迟）、pandas 输出 CSV，并做了 **top-k（2/3/5）对照实验**（[evaluation/evaluate.py](evaluation/evaluate.py), [evaluation/run_experiments.py](evaluation/run_experiments.py)）。
- 单元测试覆盖分块、检索、服务编排、API 四个层次，**核心逻辑测试不依赖外部 API**（用 Fake 对象隔离 OpenAI 依赖）（[tests/](tests/)）。
- 在真实 API 上跑通过一次完整评测，记录了具体的延迟数字（约 1.4–2.4 秒端到端）和 100% 的检索/关键词命中率（数据集仅 11 题，规模需要如实说明，见第 8 节）。

## 8. 未实现 / 只是部分实现的部分（明确列出，不美化）

- **没有 Docker 化，没有任何 CI/CD 流程**（无 `Dockerfile`、无 GitHub Actions 等）。README 里提到的部署方式仅限于本地 `uvicorn --reload`。
- **`.env.example` 文件实际上不存在**，但 README 的快速开始步骤写着 `cp .env.example .env`（[README.md:21](README.md:21)）——这是文档和仓库现状不一致的一个小 bug，按 README 操作会直接失败。
- **没有真正的数据一致性/并发保证**：索引是纯内存对象，仅进程启动/退出时读写文件（[app/main.py:26-27, 36](app/main.py:26)）；多个 worker 进程、并发写入 `/documents`、或进程非正常退出（如被 kill）都没有处理——没有锁、没有 WAL、没有原子写。
- **只支持 `.txt` 纯文本**，代码注释里自己写了"PDF can be a future extension"（[app/ingestion.py:52](app/ingestion.py:52)），PDF/Word/HTML 等格式解析未实现。
- **没有认证/鉴权、没有限流（rate limiting）**：任何能访问到该服务的请求都可以任意读写索引、消耗 OpenAI API 额度。
- **没有对 OpenAI API 调用的超时、重试、熔断机制**：`OpenAIEmbedder`/`OpenAIGenerator` 直接同步调用 SDK，出错时统一转成 502，没有指数退避重试。
- **没有增量更新/删除文档的能力**：`FaissVectorStore` 只有 `add()`，没有按 chunk id 删除或更新的方法；要清空索引只能删掉 `data/` 目录下的文件。
- **没有多轮对话/会话历史**：`/ask` 是无状态的单轮问答，不支持基于上下文的追问。
- **评测规模非常小**（11 条问题、2 份示例文档），README 自己也强调 100% 的指标不能说明 top-k 参数不影响效果，只能说明这套数据集区分度不够——**不要在简历里把"100% 准确率"当作可泛化的效果结论来写**，应如实说明是"11 题小型评测集上的结果"。
- **`faithfulness_proxy` 和 `answer_keyword_match` 都是关键词匹配的简单代理指标**，不是语义级别的正确性/忠实度评估（如 LLM-as-judge 或人工标注），代码注释里也说明了这一点（[evaluation/evaluate.py:44-46](evaluation/evaluate.py:44)）。
- **没有向量数据库层面的扩展能力**：`IndexFlatIP` 是精确暴力搜索，数据规模变大（比如百万级 chunk）后检索延迟会线性增长，没有引入近似最近邻索引或外部向量数据库（如 Milvus、pgvector、Pinecone）。

## 9. 面试中可能被追问的技术点

- **"为什么用内积索引而不是 L2 距离索引？"** —— 答案在 [app/retrieval.py:1, 13-18](app/retrieval.py:1)：因为向量已做 L2 归一化，内积在数学上等价于 cosine similarity；如果不归一化，`IndexFlatIP` 得到的排序会同时受向量模长影响，不再是纯语义相似度。
- **"如果两个 worker 进程同时收到 `/documents` 请求会发生什么？"** —— 会发生数据竞争：每个进程各自维护自己的内存索引和 `data/` 文件写入，后写入的会覆盖先写入的，这是当前架构没有解决的真实缺陷（第 8 节已列出）。可以借此展开讲你会如何改造成外部向量数据库 + 数据库锁/乐观并发控制。
- **"分块策略的局限是什么？"** —— 纯字符数切分对结构化文本（代码、表格、列表）不友好，长句可能被从语义边界切开；可以谈到替代方案（按句子/段落/token 数分块、递归字符分块、语义分块）。
- **"grounding prompt 真的能防止幻觉吗？"** —— 不能完全防止，只能降低概率；模型可能不完全服从指令、检索本身可能召回错误 chunk、文档本身可能有错误信息。可以谈评测里 `faithfulness_proxy` 只是关键词代理，真正的忠实度评估需要人工审核或更强的 LLM-as-judge。
- **"top_k 增大对系统的影响？"** —— 更高召回率的同时会扩大 prompt 长度（更高 token 成本、可能触发上下文长度限制）、增加噪声（不相关 chunk 可能干扰模型判断）、增加延迟。README 中的对照实验数据可以直接作为讨论素材，但要诚实说明这套小数据集看不出质量差异。
- **"延迟 2395ms 主要花在哪？"** —— 代码里的 `latency_ms` 只统计了 query embedding + 检索 + LLM 生成这一段（[app/generation.py:60-71](app/generation.py:60)），没有单独拆分 embedding 调用耗时和 chat 生成耗时；如果被追问"能不能优化延迟"，可以谈流式返回（streaming）、embedding 缓存、异步并发处理多个请求等改进方向，这些目前都未实现。
- **"这个系统能力的边界是什么，能撑住多大数据量/并发？"** —— 单进程、内存索引、精确暴力搜索，适合演示级别的小数据集（当前示例文档共约 292 词，见 `documents/*.txt` 的 `wc -w` 结果）；不适合生产级并发或大规模文档库，需要在第 8 节列出的几个方向上重新设计（外部向量库、异步任务队列、认证鉴权、可观测性）。

---

补充参考：更完整的中文面试问答讲解见 [INTERVIEW_GUIDE.zh-CN.md](INTERVIEW_GUIDE.zh-CN.md)（本文档未逐条核对该文件内容，如需要可另行核查）。本文档未包含任何 API key、密码或密钥；`.env` 文件内容未被读取或引用。
