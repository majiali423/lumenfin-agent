from __future__ import annotations

from dataclasses import dataclass

from ..config import AppConfig
from ..llm import BaseLLMClient, build_llm_client
from ..market_data import MarketDataClient
from ..rag.embeddings import build_embedding_provider
from ..rag.factory import build_hybrid_retriever
from .adapters import (
    EmbeddingProviderAdapter,
    LLMProviderAdapter,
    MarketDataProviderAdapter,
    RetrieverProviderAdapter,
)


@dataclass(frozen=True)
class ProviderRegistry:
    llm: LLMProviderAdapter
    market_data: MarketDataProviderAdapter
    retriever: RetrieverProviderAdapter
    embedding: EmbeddingProviderAdapter

    def health_report(self) -> dict[str, dict[str, object]]:
        providers = {
            "llm": self.llm,
            "market_data": self.market_data,
            "retriever": self.retriever,
            "embedding": self.embedding,
        }
        return {
            name: {
                "ok": provider.health_check().ok,
                "name": provider.health_check().name,
                "mode": provider.health_check().mode,
                "detail": provider.health_check().detail,
                "is_mock": provider.is_mock,
            }
            for name, provider in providers.items()
        }


def build_provider_registry(
    config: AppConfig,
    *,
    llm_client: BaseLLMClient | None = None,
    market_data_client: MarketDataClient | None = None,
) -> ProviderRegistry:
    llm = llm_client or build_llm_client(config.llm)
    market = market_data_client or MarketDataClient(
        provider=config.market_data_provider,
        alphavantage_api_key=config.alphavantage_api_key,
    )
    retriever = build_hybrid_retriever(config)
    embedder = build_embedding_provider(config.embedding_provider, config.embedding_dimension)
    market_mode = "mock" if config.market_data_provider in {"fake", "mock"} else "live"
    return ProviderRegistry(
        llm=LLMProviderAdapter(llm),
        market_data=MarketDataProviderAdapter(market, mode=market_mode),
        retriever=RetrieverProviderAdapter(retriever),
        embedding=EmbeddingProviderAdapter(embedder, name=config.embedding_provider),
    )
