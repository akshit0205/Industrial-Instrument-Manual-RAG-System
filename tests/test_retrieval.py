import numpy as np

from src.schemas import ChunkMetadata
from src.vector_store import VectorStore


class FakeModel:
    def encode(self, values, normalize_embeddings=True):
        vectors = []
        for value in values:
            vectors.append([1.0, 0.0] if "pressure" in value.lower() else [0.0, 1.0])
        return np.array(vectors, dtype="float32")


def test_index_roundtrip_and_top_k(monkeypatch, tmp_path):
    monkeypatch.setattr("src.vector_store.get_model", lambda: FakeModel())
    chunks = [
        ChunkMetadata(chunk_id="a", source="a.pdf", page=1, text="pressure limit"),
        ChunkMetadata(chunk_id="b", source="b.pdf", page=2, text="temperature range"),
    ]
    store = VectorStore()
    store.build(chunks)
    store.save(tmp_path)
    loaded = VectorStore.load(tmp_path)
    results = loaded.retrieve("pressure setting", 5)
    assert len(results) == 2
    assert results[0][0].source == "a.pdf"
    assert results[0][1] >= results[1][1]
