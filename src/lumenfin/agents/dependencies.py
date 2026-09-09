"""Explicit runtime services LangGraph specialist mixins may use.

Mixins still compose on ``AgentRuntime``; this Protocol makes the shared
attributes and step helpers visible instead of relying on implicit mixin fields.
``FinanceState`` remains a LangGraph-compatible dict at the graph boundary.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol

from ..input_guardrail import GuardrailMode
from ..knowledge_store import KnowledgeStore
from ..llm import BaseLLMClient
from ..market_data import MarketDataClient
from ..memory import ReasoningMemory, SessionMemory
from ..observability import StepTimer
from ..rag.hybrid_retriever import HybridEvidenceRetriever
from ..state import FinanceState


class RuntimeDependencies(Protocol):
    session_memory: SessionMemory
    knowledge_memory: KnowledgeStore
    reasoning_memory: ReasoningMemory
    llm_client: BaseLLMClient
    market_data_client: MarketDataClient
    hybrid_retriever: HybridEvidenceRetriever | None
    rag_enabled: bool
    rag_index_mode: str
    company_parallelism: int
    profile_llm_max_attempts: int
    input_guardrail_enabled: bool
    input_guardrail_mode: GuardrailMode
    rag_sanitize_hits: bool
    tool_backend: str
    allow_sample_data: bool
    data_mode: str
    fetch_live_fundamentals: bool
    fetch_sec_fundamentals: bool
    task_spec_gating: bool
    bounded_repair_enabled: bool
    bounded_repair_deadline_seconds: float
    bounded_repair_max_steps: int
    bounded_repair_max_tool_calls: int

    def _record(
        self,
        step: str,
        status: str,
        detail: str,
        state: FinanceState,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...

    def _track_step(self, step: str) -> Iterator[StepTimer]: ...
