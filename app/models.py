from __future__ import annotations

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    chunk_index: int


class DocumentInput(BaseModel):
    filename: str = Field(min_length=1, max_length=200)
    text: str = Field(min_length=1)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class RetrievedChunk(Chunk):
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: float

# Response DTO
class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int
