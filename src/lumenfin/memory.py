from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from .state import AuditEvent


@dataclass
class SessionMemory:
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def save(self, state: dict[str, Any]) -> None:
        self.checkpoints.append(dict(state))

    def latest(self) -> dict[str, Any]:
        return self.checkpoints[-1] if self.checkpoints else {}


@dataclass
class KnowledgeGraphMemory:
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)

    def ingest_company_document(self, company: str, payload: dict[str, Any]) -> None:
        self.graph.add_node(company, kind="company")
        for metric_name, value in payload.get("market_data", {}).items():
            metric_id = f"{company}:{metric_name}"
            self.graph.add_node(metric_id, kind="metric", value=value)
            self.graph.add_edge(company, metric_id, relation="HAS_METRIC")

        supply = payload.get("supply_chain", {})
        risk_id = f"{company}:supply_chain_risk"
        self.graph.add_node(risk_id, kind="risk", level=supply.get("risk_level", "unknown"))
        self.graph.add_edge(company, risk_id, relation="HAS_RISK")

        appendix = payload.get("appendix", {})
        for field_name, value in appendix.items():
            appendix_id = f"{company}:appendix:{field_name}"
            self.graph.add_node(appendix_id, kind="appendix", value=value)
            self.graph.add_edge(company, appendix_id, relation="HAS_APPENDIX_ITEM")

    def snapshot(self) -> dict[str, Any]:
        nodes = []
        for node, attrs in self.graph.nodes(data=True):
            nodes.append({"id": node, **attrs})
        edges = []
        for source, target, attrs in self.graph.edges(data=True):
            edges.append({"source": source, "target": target, **attrs})
        return {"nodes": nodes, "edges": edges}


@dataclass
class ReasoningMemory:
    events: list[AuditEvent] = field(default_factory=list)

    def record(self, step: str, status: str, detail: str, **metrics: Any) -> AuditEvent:
        event: AuditEvent = {"step": step, "status": status, "detail": detail}
        for key in (
            "started_at",
            "ended_at",
            "latency_ms",
            "model",
            "prompt_tokens",
            "completion_tokens",
            "estimated_cost_usd",
            "tool_calls",
            "retry_count",
        ):
            if key in metrics and metrics[key] is not None:
                event[key] = metrics[key]  # type: ignore[literal-required]
        self.events.append(event)
        return event

    def export(self) -> list[AuditEvent]:
        return list(self.events)
