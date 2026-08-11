from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import networkx as nx


class KnowledgeStore(Protocol):
    def ingest_company_document(self, company: str, payload: dict[str, Any]) -> None: ...

    def snapshot(self) -> dict[str, Any]: ...


@dataclass
class InMemoryKnowledgeStore:
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

        for doc in payload.get("source_documents", []):
            doc_id = f"{company}:document:{doc.get('document_id', doc.get('filename', 'unknown'))}"
            self.graph.add_node(doc_id, kind="document", filename=doc.get("filename", "unknown"))
            self.graph.add_edge(company, doc_id, relation="HAS_DOCUMENT")

    def snapshot(self) -> dict[str, Any]:
        nodes = [{"id": node, **attrs} for node, attrs in self.graph.nodes(data=True)]
        edges = [{"source": source, "target": target, **attrs} for source, target, attrs in self.graph.edges(data=True)]
        return {"backend": "networkx", "nodes": nodes, "edges": edges}
