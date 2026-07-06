from __future__ import annotations

from typing import Any

from ..llm import BaseLLMClient
from ..market_data import MarketDataClient
from ..rag.embeddings import EmbeddingProvider
from ..rag.hybrid_retriever import HybridEvidenceRetriever
from .base import LLMProvider, MarketDataProvider, ProviderHealth


class LLMProviderAdapter(LLMProvider):
    def __init__(self, client: BaseLLMClient, *, mode: str | None = None) -> None:
        self._client = client
        self._mode = mode or ("mock" if client.backend_name in {"local-fallback", "fake"} else "live")

    @property
    def client(self) -> BaseLLMClient:
        return self._client

    @property
    def name(self) -> str:
        return self._client.backend_name

    @property
    def is_mock(self) -> bool:
        return self._mode == "mock"

    def health_check(self) -> ProviderHealth:
        if self._client.backend_name == "local-fallback":
            return ProviderHealth(
                ok=True,
                name=self.name,
                mode="mock",
                detail="Deterministic local fallback LLM is active.",
            )
        if self._client.backend_name == "deepseek":
            return ProviderHealth(
                ok=True,
                name=self.name,
                mode="live",
                detail="DeepSeek chat backend configured.",
            )
        return ProviderHealth(ok=True, name=self.name, mode=self._mode, detail="LLM provider reachable.")

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str:
        return self._client.chat(
            system_prompt,
            user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )


class MarketDataProviderAdapter(MarketDataProvider):
    def __init__(self, client: MarketDataClient, *, mode: str | None = None) -> None:
        self._client = client
        self._mode = mode or ("mock" if client.provider in {"fake", "mock"} else "live")

    @property
    def client(self) -> MarketDataClient:
        return self._client

    @property
    def name(self) -> str:
        return self._client.provider

    @property
    def is_mock(self) -> bool:
        return self._mode == "mock"

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True,
            name=self.name,
            mode=self._mode,
            detail=f"Market data provider={self.name}.",
        )

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        return self._client.fetch_company_snapshot(company, symbol=symbol)


class RetrieverProviderAdapter:
    def __init__(self, retriever: HybridEvidenceRetriever | None, *, name: str = "hybrid-milvus") -> None:
        self._retriever = retriever
        self._name = name

    @property
    def name(self) -> str:
        return self._name if self._retriever is not None else "disabled"

    @property
    def is_mock(self) -> bool:
        return self._retriever is None

    def health_check(self) -> ProviderHealth:
        if self._retriever is None:
            return ProviderHealth(ok=True, name=self.name, mode="mock", detail="Hybrid retriever disabled.")
        return ProviderHealth(ok=True, name=self.name, mode="live", detail="Hybrid Milvus retriever enabled.")

    def search(
        self,
        *,
        company: str,
        query: str,
        document_contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if self._retriever is None:
            return []
        return self._retriever.search(
            company=company,
            query=query,
            document_contexts=document_contexts,
        )


class EmbeddingProviderAdapter:
    def __init__(self, provider: EmbeddingProvider, *, name: str) -> None:
        self._provider = provider
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_mock(self) -> bool:
        return self._name == "deterministic"

    @property
    def dimension(self) -> int:
        return self._provider.dimension

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            ok=True,
            name=self.name,
            mode="mock" if self.is_mock else "live",
            detail=f"Embedding dimension={self.dimension}.",
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._provider.embed(texts)
