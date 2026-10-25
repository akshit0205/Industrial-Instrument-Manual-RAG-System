"""Normalized SentenceTransformer embeddings backed by FAISS."""
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import settings
from .schemas import ChunkMetadata

_model: Any | None = None


def _faiss() -> Any:
    """Import FAISS only in operations that require the native extension."""
    import faiss
    return faiss


def get_model() -> Any:
    global _model
    if _model is None:
        # Delay the heavyweight Torch import until actual embeddings are requested.
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(settings.model)
    return _model


class VectorStore:
    def __init__(self, index: Any | None = None, metadata: list[ChunkMetadata] | None = None):
        self.index = index
        self.metadata = metadata or []

    @property
    def size(self) -> int:
        return len(self.metadata)

    def build(self, chunks: list[ChunkMetadata]) -> None:
        self.metadata = chunks
        if not chunks:
            self.index = None
            return
        vectors = np.asarray(get_model().encode(
            [chunk.text for chunk in chunks], normalize_embeddings=True
        ), dtype="float32")
        faiss = _faiss()
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump([item.model_dump() for item in self.metadata], file, indent=2)
        if self.index is not None:
            _faiss().write_index(self.index, str(target / "faiss.index"))

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        target = Path(directory)
        metadata_path = target / "metadata.json"
        index_path = target / "faiss.index"
        if not metadata_path.exists() or not index_path.exists():
            return cls()
        with metadata_path.open(encoding="utf-8") as file:
            metadata = [ChunkMetadata.model_validate(item) for item in json.load(file)]
        return cls(_faiss().read_index(str(index_path)), metadata)

    def retrieve(self, query: str, top_k: int) -> list[tuple[ChunkMetadata, float]]:
        if self.index is None or not self.metadata:
            return []
        q_vec = np.asarray(get_model().encode([query], normalize_embeddings=True), dtype="float32")
        scores, indices = self.index.search(q_vec, min(top_k, self.index.ntotal))
        return [
            (self.metadata[index], float(score))
            for index, score in zip(indices[0], scores[0])
            if index >= 0
        ]
