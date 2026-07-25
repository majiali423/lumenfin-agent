from __future__ import annotations

import hashlib
import logging
import math
import os
import re
import time
from typing import Any, Callable, Protocol

import httpx

from ..provider_retry import call_with_transient_retry

logger = logging.getLogger(__name__)


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


# text-embedding-v3 compatible dimensions (v4 allows a superset; we accept common set).
_DASHSCOPE_DIMS = frozenset({64, 128, 256, 512, 768, 1024, 1536, 2048, 2560})
_DASHSCOPE_BATCH = 10  # v3/v4 list input cap


class DashScopeEmbeddingProvider:
    """Alibaba Cloud DashScope text embeddings (OpenAI-compatible HTTP API).

    Docs: https://help.aliyun.com/zh/model-studio/text-embedding-synchronous-api
    API key: https://bailian.console.aliyun.com/ → API-KEY
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int = 1024,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        key = (api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY") or "").strip()
        if not key:
            raise ValueError(
                "DASHSCOPE_API_KEY is required when MAS_EMBEDDING_PROVIDER=dashscope. "
                "Create a key at https://bailian.console.aliyun.com/ (API-KEY)."
            )
        dim = int(dimension)
        if dim not in _DASHSCOPE_DIMS:
            raise ValueError(
                f"Unsupported DashScope embedding dimension {dim}; "
                f"choose one of {sorted(_DASHSCOPE_DIMS)}."
            )
        self._api_key = key
        self._model = (
            model
            or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
            or "text-embedding-v3"
        ).strip()
        self._dimension = dim
        self._base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self._timeout = float(timeout_seconds)
        self._client = client

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _DASHSCOPE_BATCH):
            batch = texts[start : start + _DASHSCOPE_BATCH]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimension,
        }

        def _post() -> httpx.Response:
            if self._client is not None:
                return self._client.post(url, headers=headers, json=payload)
            with httpx.Client(timeout=self._timeout) as client:
                return client.post(url, headers=headers, json=payload)

        response = _post()
        response.raise_for_status()
        data = response.json()
        items = data.get("data") or []
        # OpenAI-compatible responses may be unordered; sort by index.
        ordered = sorted(items, key=lambda item: int(item.get("index", 0)))
        vectors = [list(map(float, item.get("embedding") or [])) for item in ordered]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"DashScope embedding count mismatch: got {len(vectors)} for {len(texts)} inputs."
            )
        for vector in vectors:
            if len(vector) != self._dimension:
                raise RuntimeError(
                    f"DashScope embedding dimension mismatch: expected {self._dimension}, "
                    f"got {len(vector)}. Recreate the Milvus collection after changing models."
                )
        return vectors


class ResilientEmbeddingProvider:
    """Wrap any embedder with transient retry (429/5xx/timeout/connection)."""

    def __init__(
        self,
        inner: EmbeddingProvider,
        *,
        max_retries: int = 3,
        backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._inner = inner
        self.max_retries = max(1, int(max_retries))
        self.backoff_seconds = float(backoff_seconds)
        self._sleep = sleep
        self.last_attempts = 0
        self.last_error: str | None = None
        self.last_embed_ms = 0.0
        self.last_embed_chars = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def inner(self) -> EmbeddingProvider:
        return self._inner

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.last_attempts = 0
        self.last_error = None
        self.last_embed_ms = 0.0
        self.last_embed_chars = sum(len(text or "") for text in texts)
        attempts = {"n": 0}
        started = time.perf_counter()

        def _call() -> list[list[float]]:
            attempts["n"] += 1
            return self._inner.embed(texts)

        try:
            vectors = call_with_transient_retry(
                _call,
                max_retries=self.max_retries,
                backoff_seconds=self.backoff_seconds,
                sleep=self._sleep,
            )
            self.last_attempts = attempts["n"]
            self.last_embed_ms = round((time.perf_counter() - started) * 1000, 2)
            return vectors
        except Exception as exc:
            self.last_attempts = attempts["n"]
            self.last_embed_ms = round((time.perf_counter() - started) * 1000, 2)
            self.last_error = str(exc)
            logger.warning(
                "Embedding failed after %s attempt(s): %s",
                attempts["n"],
                exc,
            )
            raise


def build_embedding_provider(
    provider_name: str,
    dimension: int = 384,
    *,
    max_retries: int | None = None,
    backoff_seconds: float | None = None,
    timeout_seconds: float | None = None,
    resilient: bool | None = None,
) -> EmbeddingProvider:
    normalized = provider_name.strip().lower()
    retries = int(max_retries if max_retries is not None else os.getenv("MAS_EMBEDDING_MAX_RETRIES", "3"))
    backoff = float(
        backoff_seconds
        if backoff_seconds is not None
        else os.getenv("MAS_EMBEDDING_BACKOFF_SECONDS", "0.5")
    )
    timeout = float(
        timeout_seconds
        if timeout_seconds is not None
        else os.getenv("DASHSCOPE_EMBEDDING_TIMEOUT")
        or os.getenv("MAS_EMBEDDING_TIMEOUT_SECONDS", "60")
    )
    wrap = resilient
    if wrap is None:
        wrap = normalized not in {"deterministic", "hash", "local"}

    if normalized == "fastembed":
        inner: EmbeddingProvider = FastEmbedProvider()
    elif normalized in {"dashscope", "aliyun", "alibaba", "通义"}:
        raw_dim = os.getenv("DASHSCOPE_EMBEDDING_DIMENSION") or str(dimension or 1024)
        inner = DashScopeEmbeddingProvider(dimension=int(raw_dim), timeout_seconds=timeout)
    else:
        inner = DeterministicEmbeddingProvider(dimension=dimension)

    if wrap:
        return ResilientEmbeddingProvider(inner, max_retries=retries, backoff_seconds=backoff)
    return inner
