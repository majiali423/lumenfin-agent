from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # pragma: no cover
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

from .clarification import merge_clarification_into_query
from .config import AppConfig
from .agents import AgentRuntime
from .planning import build_query_plan
from .skills import get_skill_specs
from .knowledge_store import InMemoryKnowledgeStore, Neo4jKnowledgeStore
from .llm import BaseLLMClient, build_llm_client
from .market_data import MarketDataClient
from .memory import ReasoningMemory, SessionMemory
from .rag.factory import build_document_indexer, build_hybrid_retriever, build_rag_store
from .state import FinanceState
from .checkpoint_store import WorkflowCheckpointRepository
from .observability import utc_now_iso


def route_after_input_guardrail(state: FinanceState) -> str:
    if state.get("workflow_status") == "blocked_by_guardrail":
        return "end"
    return "query_planner"


def route_after_query_planner(state: FinanceState) -> str:
    # Re-pause whenever planner still reports missing fields, even after a partial clarification.
    if state.get("missing_fields"):
        return "await_clarification"
    return "supervisor"


def route_after_retrieval(state: FinanceState) -> str:
    if state.get("fatal_data_gap"):
        return "claim_binder"
    return "appendix_replan" if state.get("replan_reason") else "quant"


def route_after_quant(state: FinanceState) -> str:
    return "appendix_replan" if state.get("replan_reason") else "psychologist"


def route_after_critic(state: FinanceState) -> str:
    findings = state.get("compliance_findings", [])
    if not findings:
        return "claim_binder"
    iterations = state.get("critic_iterations", 0)
    max_iterations = state.get("critic_max_iterations", 2)
    if iterations >= max_iterations:
        return "claim_binder"
    return "repair"


def route_after_repair(state: FinanceState) -> str:
    return state.get("critic_repair_target", "quant")


def route_after_appendix_replan(state: FinanceState) -> str:
    """After supplementary_retrieval planning: retry retrieval once, else synthesize."""
    return "claim_binder" if state.get("degraded_mode") else "retrieval"


# Backward-compatible alias for older imports/tests.
route_after_replanner = route_after_appendix_replan


def _base_initial_state(
    query: str,
    thread_id: str,
    document_contexts: list[dict[str, Any]] | None,
    app_config: AppConfig,
    *,
    rag_index_stats: dict[str, Any] | None = None,
    rag_document_ids: list[str] | None = None,
    rag_tenant_id: str | None = None,
) -> FinanceState:
    return {
        "query": query,
        "thread_id": thread_id,
        "document_contexts": document_contexts or [],
        "rag_document_ids": list(rag_document_ids or []),
        "rag_tenant_id": rag_tenant_id or app_config.rag_tenant_id,
        "audit_log": [],
        "reasoning_memory": [],
        "compliance_findings": [],
        "compliance_violations": [],
        "retrieval_provenance": {},
        "report_sections": [],
        "appendix_search_done": False,
        "retries": 0,
        "degraded_mode": False,
        "fatal_data_gap": False,
        "partial_data_gap": False,
        "data_gap_detail": "",
        "coverage_matrix": {},
        "non_comparable_companies": [],
        "rag_evidence": {},
        "rag_index_stats": dict(rag_index_stats or {}),
        "critic_iterations": 0,
        "critic_max_iterations": app_config.critic_max_iterations,
        "critic_repair_target": "quant",
        "workflow_status": "running",
        "user_clarification": {},
        "run_telemetry": {},
        "run_started_at": utc_now_iso(),
        "input_guardrail_findings": [],
        "input_guardrail_summary": {},
        "data_mode": app_config.data_mode,
    }


class LumenFinAgentSystem:
    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        app_config: AppConfig | None = None,
        market_data_client: MarketDataClient | None = None,
        *,
        rag_store: Any | None = None,
        document_indexer: Any | None = None,
    ) -> None:
        self.app_config = app_config or AppConfig.from_env()
        self.session_memory = SessionMemory()
        self.knowledge_memory = self._build_knowledge_store()
        self.reasoning_memory = ReasoningMemory()
        self.llm_client = llm_client or build_llm_client(
            self.app_config.llm,
            allow_local_fallback=self.app_config.allows_local_fallback(),
        )
        self.market_data_client = market_data_client or MarketDataClient(
            provider=self.app_config.market_data_provider,
            alphavantage_api_key=self.app_config.alphavantage_api_key,
            fallback_provider=self.app_config.market_data_fallback,
            cache_ttl_seconds=self.app_config.market_cache_ttl_seconds,
        )
        self.rag_store = rag_store if rag_store is not None else build_rag_store(self.app_config)
        self.document_indexer = (
            document_indexer
            if document_indexer is not None
            else build_document_indexer(self.app_config, rag_store=self.rag_store)
        )
        hybrid_retriever = build_hybrid_retriever(
            self.app_config,
            rag_store=self.rag_store,
            indexer=self.document_indexer,
        )
        self.runtime = AgentRuntime(
            session_memory=self.session_memory,
            knowledge_memory=self.knowledge_memory,
            reasoning_memory=self.reasoning_memory,
            llm_client=self.llm_client,
            market_data_client=self.market_data_client,
            hybrid_retriever=hybrid_retriever,
            rag_enabled=self.app_config.rag_enabled,
            rag_index_mode=self.app_config.rag_index_mode,
            company_parallelism=self.app_config.company_parallelism,
            profile_llm_max_attempts=self.app_config.profile_llm_max_attempts,
            input_guardrail_enabled=self.app_config.input_guardrail_enabled,
            input_guardrail_mode=self.app_config.input_guardrail_mode,  # type: ignore[arg-type]
            rag_sanitize_hits=self.app_config.rag_sanitize_hits,
            tool_backend=self.app_config.tool_backend,
            allow_sample_data=self.app_config.allows_sample_data(),
            data_mode=self.app_config.data_mode,
            fetch_live_fundamentals=self.app_config.fetch_live_fundamentals,
            fetch_sec_fundamentals=self.app_config.fetch_sec_fundamentals,
        )
        self.checkpointer = InMemorySaver()
        self.graph = self._build_graph()

    def _build_knowledge_store(self):
        if self.app_config.neo4j_uri and self.app_config.neo4j_username and self.app_config.neo4j_password:
            try:
                return Neo4jKnowledgeStore(
                    uri=self.app_config.neo4j_uri,
                    username=self.app_config.neo4j_username,
                    password=self.app_config.neo4j_password,
                )
            except Exception:
                return InMemoryKnowledgeStore()
        return InMemoryKnowledgeStore()

    def _build_graph(self):
        workflow = StateGraph(FinanceState)
        workflow.add_node("input_guardrail", self.runtime.input_guardrail)
        workflow.add_node("query_planner", self.runtime.query_planner)
        workflow.add_node("await_clarification", self.runtime.await_clarification)
        workflow.add_node("supervisor", self.runtime.supervisor)
        workflow.add_node("retrieval", self.runtime.retrieval)
        workflow.add_node("quant", self.runtime.quantitative_analyst)
        workflow.add_node("psychologist", self.runtime.psychologist)
        workflow.add_node("critic", self.runtime.critic)
        workflow.add_node("repair", self.runtime.repair)
        workflow.add_node("appendix_replan", self.runtime.appendix_replan)
        workflow.add_node("claim_binder", self.runtime.claim_binder)
        workflow.add_node("synthesizer", self.runtime.synthesizer)

        workflow.add_edge(START, "input_guardrail")
        workflow.add_conditional_edges(
            "input_guardrail",
            route_after_input_guardrail,
            {"query_planner": "query_planner", "end": END},
        )
        workflow.add_conditional_edges(
            "query_planner",
            route_after_query_planner,
            {"supervisor": "supervisor", "await_clarification": "await_clarification"},
        )
        workflow.add_edge("await_clarification", END)
        workflow.add_edge("supervisor", "retrieval")
        workflow.add_conditional_edges(
            "retrieval",
            route_after_retrieval,
            {"quant": "quant", "appendix_replan": "appendix_replan", "claim_binder": "claim_binder"},
        )
        workflow.add_conditional_edges(
            "quant",
            route_after_quant,
            {"psychologist": "psychologist", "appendix_replan": "appendix_replan"},
        )
        workflow.add_edge("psychologist", "critic")
        workflow.add_conditional_edges(
            "critic",
            route_after_critic,
            {"claim_binder": "claim_binder", "repair": "repair"},
        )
        workflow.add_conditional_edges(
            "repair",
            route_after_repair,
            {"retrieval": "retrieval", "quant": "quant", "psychologist": "psychologist"},
        )
        workflow.add_conditional_edges(
            "appendix_replan",
            route_after_appendix_replan,
            {"retrieval": "retrieval", "claim_binder": "claim_binder"},
        )
        workflow.add_edge("claim_binder", "synthesizer")
        workflow.add_edge("synthesizer", END)

        return workflow.compile(checkpointer=self.checkpointer)

    def run(
        self,
        query: str,
        thread_id: str = "demo-thread",
        document_contexts: list[dict[str, Any]] | None = None,
        *,
        rag_index_stats: dict[str, Any] | None = None,
        rag_document_ids: list[str] | None = None,
        rag_tenant_id: str | None = None,
    ) -> dict[str, Any]:
        self.reasoning_memory.events.clear()
        initial_state = _base_initial_state(
            query,
            thread_id,
            document_contexts,
            self.app_config,
            rag_index_stats=rag_index_stats,
            rag_document_ids=rag_document_ids,
            rag_tenant_id=rag_tenant_id,
        )
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(initial_state, config=config)
        return self._finalize_result(result, thread_id)

    def resume_with_clarification(
        self,
        thread_id: str,
        clarification: dict[str, Any],
    ) -> dict[str, Any]:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        if snapshot is None or snapshot.values is None:
            raise ValueError(f"Thread {thread_id} not found.")

        current = dict(snapshot.values)
        effective_query = merge_clarification_into_query(current.get("query", ""), clarification)
        query_plan = build_query_plan(
            effective_query,
            document_contexts=current.get("document_contexts", []),
            llm_client=self.llm_client,
            user_clarification=clarification,
        )
        required_skills = query_plan.required_skills
        self.graph.update_state(
            config,
            {
                "user_clarification": clarification,
                "workflow_status": "running",
                "query": query_plan.normalized_query,
                "query_plan": query_plan.to_dict(),
                "required_skills": required_skills,
                "skill_specs": get_skill_specs(required_skills),
                "missing_fields": query_plan.missing_fields,
                "clarification_questions": query_plan.clarification_questions,
            },
            as_node="query_planner",
        )
        result = self.graph.invoke(None, config=config)
        return self._finalize_result(result, thread_id)

    def get_thread_state(self, thread_id: str) -> dict[str, Any] | None:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        if snapshot is None or snapshot.values is None:
            return None
        values = dict(snapshot.values)
        if not values.get("query") and not values.get("workflow_status"):
            return None
        return values

    def bootstrap_thread_from_store(
        self,
        thread_id: str,
        checkpoint_repo: WorkflowCheckpointRepository,
    ) -> dict[str, Any]:
        record = checkpoint_repo.get(thread_id)
        if record is None:
            raise ValueError(f"No persisted checkpoint found for thread_id={thread_id}")
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = self.graph.get_state(config)
        if snapshot is not None and snapshot.values:
            return dict(snapshot.values)
        state = dict(record["state"])
        self.graph.update_state(config, state, as_node=record["last_node"])
        return state

    def persist_checkpoint(
        self,
        thread_id: str,
        query: str,
        result: dict[str, Any],
        checkpoint_repo: WorkflowCheckpointRepository,
    ) -> dict[str, Any]:
        state = self.get_thread_state(thread_id) or result
        merged = {**state, **{k: v for k, v in result.items() if k not in state or v is not None}}
        return checkpoint_repo.upsert(
            thread_id=thread_id,
            query=query,
            state=merged,
            llm_backend=result.get("llm_backend", self.llm_client.backend_name),
        )

    def _finalize_result(self, result: dict[str, Any], thread_id: str) -> dict[str, Any]:
        final_result = dict(result)
        final_result["thread_id"] = thread_id
        final_result["checkpoint_count"] = len(self.session_memory.checkpoints)
        final_result["llm_backend"] = self.llm_client.backend_name
        final_result.setdefault("data_mode", self.app_config.data_mode)
        final_result.setdefault("run_started_at", utc_now_iso())
        final_result["run_ended_at"] = utc_now_iso()
        if final_result.get("workflow_status") == "needs_clarification":
            final_result.setdefault("final_report", "")
        if final_result.get("workflow_status") == "blocked_by_guardrail":
            final_result.setdefault("final_report", "")
        if final_result.get("workflow_status") == "incomplete_data":
            final_result.setdefault("final_report", "")
        return final_result
