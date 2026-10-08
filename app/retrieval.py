"""FAISS vector storage. Normalized vectors + inner product equals cosine similarity."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from app.models import Chunk, RetrievedChunk


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32).copy()
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    faiss.normalize_L2(vectors)
    return vectors


class FaissVectorStore:
    def __init__(self) -> None:
        # FAISS 向量索引：按加入顺序保存每个 chunk 的 embedding，并负责 top-k 相似度检索。
        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[Chunk] = []

    @property
    def size(self) -> int: # 这样就可以直接 e.g store.size而不用store.size()
        return len(self.chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("Each chunk must have exactly one embedding")
        vectors = normalize_vectors(embeddings)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        if vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimensions do not match the existing index")
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        if self.index is None or not self.chunks:
            return []
        query = normalize_vectors(np.asarray(query_embedding, dtype=np.float32).reshape(1, -1))
        scores, positions = self.index.search(query, min(top_k, self.size))
        return [
            RetrievedChunk(**self.chunks[position].model_dump(), score=float(score))
            for score, position in zip(scores[0], positions[0])
            if position != -1
        ]

    def save(self, index_path: Path, metadata_path: Path) -> None:
        if self.index is None:
            return
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        metadata_path.write_text(json.dumps([chunk.model_dump() for chunk in self.chunks], indent=2), encoding="utf-8")

    def load(self, index_path: Path, metadata_path: Path) -> bool:
        if not index_path.exists() or not metadata_path.exists():
            return False
        self.index = faiss.read_index(str(index_path))
        self.chunks = [Chunk(**item) for item in json.loads(metadata_path.read_text(encoding="utf-8"))]
        return True


# 本模块负责向量检索：保存 chunk embeddings，按 query embedding 返回 top-k 相似 chunks，
# 并通过 save/load 将 FAISS index 与 chunk metadata 持久化到本地，避免服务重启后重新 embedding。
