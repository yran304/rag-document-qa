from __future__ import annotations

import logging
import time
from typing import Protocol

from openai import OpenAI

from app.embeddings import Embedder
from app.models import AskResponse, Chunk, RetrievedChunk
from app.retrieval import FaissVectorStore

logger = logging.getLogger(__name__)

GROUNDING_INSTRUCTIONS = """You answer questions only from the supplied document context.
If the context does not contain enough information, say exactly: "I don't have enough information in the provided documents."
Do not use outside knowledge. Cite source names in square brackets when answering.
Keep the answer concise.
"""


class Generator(Protocol):
    def generate(self, prompt: str) -> str: ...


class OpenAIGenerator:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.responses.create(model=self.model, input=prompt)
        return response.output_text.strip()

# 将检索到的 top-k 文档 chunks、用户问题和 grounding instructions 组合成完整 prompt。
# LLM 只会看到这些被检索出的上下文，而不会直接访问本地全部 documents。
def build_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    context = "\n\n".join(f"SOURCE: {item.source}\nCHUNK: {item.id}\nTEXT: {item.text}" for item in retrieved)
    return f"{GROUNDING_INSTRUCTIONS}\nDOCUMENT CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"


class RagService:
    def __init__(self, embedder: Embedder, generator: Generator, store: FaissVectorStore, default_top_k: int = 3) -> None:
        self.embedder = embedder
        self.generator = generator
        self.store = store
        self.default_top_k = default_top_k

    # 为已切分的 chunks 批量生成 embeddings，并将 chunks 与对应向量一起存入 FAISS vector store。
    def index(self, chunks: list[Chunk]) -> int:
        embeddings = self.embedder.embed([chunk.text for chunk in chunks])
        self.store.add(chunks, embeddings)
        logger.info("Indexed %s chunks; index now contains %s chunks", len(chunks), self.store.size)
        return len(chunks)

    def ask(self, question: str, top_k: int | None = None) -> AskResponse:
        if not question.strip():
            raise ValueError("Question cannot be blank")
        # 在start里存下开始前的高精度时间点，只是存当前的时间，这个不是计时器！！！
        start = time.perf_counter()
        if self.store.size == 0:
            retrieved: list[RetrievedChunk] = []
            answer = "I don't have enough information in the provided documents."
        else:
            retrieved = self.store.search(self.embedder.embed([question])[0], top_k or self.default_top_k)
            if not retrieved:
                answer = "I don't have enough information in the provided documents."
            else:
                answer = self.generator.generate(build_prompt(question, retrieved))
        # 执行完的时间 减去 开始是的时间就是耗时，这个不是计时器，所以不需要close timer()
        latency_ms = (time.perf_counter() - start) * 1000
        return AskResponse(
            answer=answer,
            sources=sorted({item.source for item in retrieved}),
            retrieved_chunks=retrieved,
            latency_ms=round(latency_ms, 2),
        )
