from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProviderHealth:
    ok: bool
    name: str
    mode: str
    detail: str = ""


class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_mock(self) -> bool: ...

    @abstractmethod
    def health_check(self) -> ProviderHealth: ...

    @abstractmethod
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 600,
    ) -> str: ...


class MarketDataProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def is_mock(self) -> bool: ...

    @abstractmethod
    def health_check(self) -> ProviderHealth: ...

    @abstractmethod
    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]: ...


class RetrieverProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_mock(self) -> bool: ...

    def health_check(self) -> ProviderHealth: ...

    def search(
        self,
        *,
        company: str,
        query: str,
        document_contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]: ...


class EmbeddingProviderInterface(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def is_mock(self) -> bool: ...

    @property
    def dimension(self) -> int: ...

    def health_check(self) -> ProviderHealth: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...
