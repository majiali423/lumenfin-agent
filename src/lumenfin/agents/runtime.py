from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from ..input_guardrail import GuardrailMode
from ..knowledge_store import KnowledgeStore
from ..llm import BaseLLMClient
from ..market_data import MarketDataClient
from ..memory import ReasoningMemory, SessionMemory
from ..observability import StepTimer, merge_telemetry
from ..rag.hybrid_retriever import HybridEvidenceRetriever
from ..state import FinanceState
from .critic import CriticMixin
from .guardrail import InputGuardrailMixin
from .planner import PlannerMixin
from .quantitative import QuantitativeMixin
from .retrieval import RetrievalMixin
from .risk import RiskMixin
from .synthesis import SynthesisMixin


class AgentRuntime(
    InputGuardrailMixin,
    PlannerMixin,
    RetrievalMixin,
    QuantitativeMixin,
    RiskMixin,
    CriticMixin,
    SynthesisMixin,
):
    def __init__(
        self,
        session_memory: SessionMemory,
        knowledge_memory: KnowledgeStore,
        reasoning_memory: ReasoningMemory,
        llm_client: BaseLLMClient,
        market_data_client: MarketDataClient,
        hybrid_retriever: HybridEvidenceRetriever | None = None,
        rag_enabled: bool = True,
        rag_index_mode: str = "sync_on_run",
        company_parallelism: int = 4,
        profile_llm_max_attempts: int = 1,
        input_guardrail_enabled: bool = True,
        input_guardrail_mode: GuardrailMode = "sanitize",
        rag_sanitize_hits: bool = True,
        tool_backend: str = "local",
        allow_sample_data: bool = True,
        data_mode: str = "demo",
        fetch_live_fundamentals: bool = False,
        fetch_sec_fundamentals: bool = False,
    ) -> None:
        self.session_memory = session_memory
        self.knowledge_memory = knowledge_memory
        self.reasoning_memory = reasoning_memory
        self.llm_client = llm_client
        self.market_data_client = market_data_client
        self.hybrid_retriever = hybrid_retriever
        self.rag_enabled = rag_enabled
        self.rag_index_mode = rag_index_mode if rag_index_mode in {"sync_on_run", "async_on_upload"} else "sync_on_run"
        self.company_parallelism = max(1, company_parallelism)
        self.profile_llm_max_attempts = max(0, min(3, int(profile_llm_max_attempts)))
        self.input_guardrail_enabled = input_guardrail_enabled
        self.input_guardrail_mode = input_guardrail_mode if input_guardrail_mode in {"sanitize", "block"} else "sanitize"
        self.rag_sanitize_hits = bool(rag_sanitize_hits)
        self.tool_backend = tool_backend if tool_backend in {"local", "mcp"} else "local"
        self.allow_sample_data = allow_sample_data
        self.data_mode = data_mode if data_mode in {"demo", "live"} else "demo"
        self.fetch_live_fundamentals = bool(fetch_live_fundamentals)
        self.fetch_sec_fundamentals = bool(fetch_sec_fundamentals)

    def _record(
        self,
        step: str,
        status: str,
        detail: str,
        state: FinanceState,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = self.reasoning_memory.record(step=step, status=status, detail=detail, **(metrics or {}))
        telemetry = merge_telemetry(state.get("run_telemetry"), event)
        return {
            "audit_log": self.reasoning_memory.export(),
            "reasoning_memory": [
                f"{item['step']}::{item['status']}::{item['detail']}" for item in self.reasoning_memory.export()
            ],
            "run_telemetry": telemetry,
        }

    @contextmanager
    def _track_step(self, step: str) -> Iterator[StepTimer]:
        self.llm_client.mark_usage_start()
        yield StepTimer(step=step, llm_client=self.llm_client)

