"""The provider boundary makes embeddings easy to inspect and fake in tests."""
from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from openai import OpenAI


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        response = self.client.embeddings.create(
            model=self.model, 
            input=list(texts), 
            encoding_format="float"
        )

        return np.asarray(
            [item.embedding for item in response.data], 
            dtype=np.float32
        )

