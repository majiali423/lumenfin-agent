from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class DeterministicEmbeddingProvider:
    """Offline-friendly embeddings for tests and no-API demos."""

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-zA-Z0-9\u4e00-\u9fff]+", text.lower())
        vector = [0.0] * self._dimension
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index in range(self._dimension):
                byte_value = digest[index % len(digest)]
                vector[index] += ((byte_value / 255.0) - 0.5) * (1.0 + (index % 7) * 0.05)
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [round(value / norm, 6) for value in vector]


class FastEmbedProvider:
    """Optional local semantic embeddings when fastembed is installed."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)
        sample = list(self._model.embed(["dimension probe"]))
        self._dimension = len(sample[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(texts)]


def build_embedding_provider(provider_name: str, dimension: int = 384) -> EmbeddingProvider:
    normalized = provider_name.strip().lower()
    if normalized == "fastembed":
        return FastEmbedProvider()
    return DeterministicEmbeddingProvider(dimension=dimension)
