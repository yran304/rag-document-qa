import numpy as np

from app.models import Chunk
from app.retrieval import FaissVectorStore


def test_search_returns_nearest_normalized_vector() -> None:
    store = FaissVectorStore()
    chunks = [
        Chunk(id="a:0", text="vacation policy", source="a.txt", chunk_index=0),
        Chunk(id="b:0", text="product launch", source="b.txt", chunk_index=0),
    ]
    store.add(chunks, np.array([[1.0, 0.0], [0.0, 3.0]], dtype=np.float32))
    result = store.search(np.array([0.1, 1.0], dtype=np.float32), top_k=1)
    assert result[0].source == "b.txt"
    assert result[0].score > 0.9

