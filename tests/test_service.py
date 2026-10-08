import numpy as np

from app.generation import RagService
from app.models import Chunk
from app.retrieval import FaissVectorStore


class FakeEmbedder:
    def embed(self, texts):
        return np.array([[1.0, 0.0] if "vacation" in text.lower() else [0.0, 1.0] for text in texts], dtype=np.float32)


class FakeGenerator:
    def generate(self, prompt: str) -> str:
        assert "DOCUMENT CONTEXT" in prompt
        return "Full-time employees receive 20 paid vacation days. [handbook.txt]"


def test_service_answers_with_retrieved_source() -> None:
    service = RagService(FakeEmbedder(), FakeGenerator(), FaissVectorStore())
    service.index([Chunk(id="h:0", text="Vacation is 20 days", source="handbook.txt", chunk_index=0)])
    answer = service.ask("How much vacation do employees get?")
    assert "20" in answer.answer
    assert answer.sources == ["handbook.txt"]

