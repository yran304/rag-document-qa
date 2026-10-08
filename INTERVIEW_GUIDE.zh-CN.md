# 面试讲解速记：这个 RAG 项目

## 30 秒版本

“我用 FastAPI 做了一个文档问答 API。上传文本后，服务会把文档按可配置的长度和重叠切成 chunks，用 embedding API 把每段转成向量，再存入 FAISS。提问时，我把问题也向量化，用 cosine similarity 取 top-k chunks，把这些原文和严格的 grounding instructions 一起放入 LLM prompt，最后返回答案、来源和分数。这样答案可以追溯到文档，而不是只靠模型参数记忆。”

## 逐步讲清楚

1. **为什么是 RAG？** LLM 不知道我的私有/最新文档。RAG 在生成前检索相关原文，把“知识更新”从重新训练模型变成重新索引文档。
2. **为什么 chunk？** 一整份文档通常太长且主题混杂。小段 embedding 的语义更集中，也避免把无关文字塞进 prompt。
3. **chunk size 和 overlap？** `chunk_size` 是单段的字符上限；`overlap` 是相邻段重复的文字。重叠避免一句话恰好被切断，但会增加 embedding、存储和重复上下文成本。
4. **embedding 是什么？** 是一串浮点数，代表文本的语义特征；相近语义通常落在向量空间相近的位置。它不是关键词列表，也不直接是“答案”。
5. **为什么 cosine similarity？** 它比较两个向量方向而不是长度。先 L2-normalize 后，向量内积就等于 cosine similarity，因此 `FAISS IndexFlatIP` 能直接做搜索。
6. **top-k 是什么？** 得分最高的 k 个 chunks。k 太小可能漏证据，太大可能加入噪声、增大 prompt 和 latency；我用评测实验而不是凭感觉选择。
7. **grounding 如何工作？** `build_prompt` 明确要求模型只能使用 `DOCUMENT CONTEXT`，不足时返回固定拒答句，同时让模型引用 source。它降低而非消灭 hallucination，因为检索与模型仍可能出错。
8. **API 流程？** `/documents` 校验输入、chunk、embed、写 index；`/ask` 校验 question、embed query、search、生成并返回 answer/sources/retrieved_chunks/latency。

## 评测怎么讲

“我写了一个 11 条问题的小型 gold dataset，每题有期望 source 和关键答案词。脚本用 pandas 记录 retrieval success、keyword answer match、faithfulness proxy 和 latency，写成 CSV。我比较了 top-k=2、3、5。这个评测不是完整自动事实裁判：keyword match 容易漏掉正确释义，faithfulness proxy 也不是 NLI；但它可复现、可解释，并能发现配置改动是否造成回归。重要问题还应该加人工抽样审查或 LLM-as-judge，并保存 prompt/model 版本。”

本次真实运行中，11/11 问题都检索到了预期来源，三个自动代理指标均为 100%，top-k=2/3/5 没有出现质量差异。这是因为两份 demo 文档很短、检索任务简单；我会如实说明这一限制，而不会把它描述成 top-k 优化带来的提升。

## 常见追问

**成本在哪里？** 每次索引调用 embedding；每次问答调用 query embedding 和生成模型。更大的 chunk、更多 documents 或更大的 top-k 会提高 token / API 成本。现版本记录 latency；生产版可从 provider usage 字段记录 input/output tokens 并按当期价格估算 cost。

**为什么没用 LangChain？** 这是有意选择。项目要展示并理解 RAG 的底层步骤；LangChain 在真实复杂工作流能提高开发效率，但会隐藏这次要学习的管道。

**目前局限？** 只支持 `.txt`；FAISS 是本地内存索引、没有多用户隔离；字符分块不是 token-aware/semantic split；评测样本小；没有 reranker。这些是下一步可逐项改进的方向，不应假装已经解决。

## 可如实使用的简历 bullet（跑通和理解后）

- Built a Python FastAPI RAG document Q&A service with direct OpenAI SDK integration, text chunking, embeddings, normalized FAISS cosine-similarity retrieval, and grounded answer/source responses.
- Created a pandas-based evaluation set and top-k retrieval experiment measuring source retrieval accuracy, keyword answer match, faithfulness proxy, and latency; exported reproducible CSV results.
