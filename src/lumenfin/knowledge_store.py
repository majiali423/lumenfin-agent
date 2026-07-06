from __future__ import annotations

from dataclasses import dataclass, field
import importlib
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


class Neo4jKnowledgeStore:
    def __init__(self, uri: str, username: str, password: str) -> None:
        neo4j_module = importlib.import_module("neo4j")
        graph_database = getattr(neo4j_module, "GraphDatabase")
        self.driver = graph_database.driver(uri, auth=(username, password))
        self._init_schema()

    def _init_schema(self) -> None:
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE")

    def ingest_company_document(self, company: str, payload: dict[str, Any]) -> None:
        with self.driver.session() as session:
            session.run("MERGE (c:Company {name: $company})", company=company)
            for metric_name, value in payload.get("market_data", {}).items():
                session.run(
                    """
                    MERGE (c:Company {name: $company})
                    MERGE (m:Metric {id: $metric_id})
                    SET m.name = $metric_name, m.value = $value
                    MERGE (c)-[:HAS_METRIC]->(m)
                    """,
                    company=company,
                    metric_id=f"{company}:{metric_name}",
                    metric_name=metric_name,
                    value=value,
                )
            supply = payload.get("supply_chain", {})
            session.run(
                """
                MERGE (c:Company {name: $company})
                MERGE (r:Risk {id: $risk_id})
                SET r.level = $level
                MERGE (c)-[:HAS_RISK]->(r)
                """,
                company=company,
                risk_id=f"{company}:supply_chain_risk",
                level=supply.get("risk_level", "unknown"),
            )
            for field_name, value in payload.get("appendix", {}).items():
                session.run(
                    """
                    MERGE (c:Company {name: $company})
                    MERGE (a:AppendixItem {id: $appendix_id})
                    SET a.name = $field_name, a.value = $value
                    MERGE (c)-[:HAS_APPENDIX_ITEM]->(a)
                    """,
                    company=company,
                    appendix_id=f"{company}:appendix:{field_name}",
                    field_name=field_name,
                    value=str(value),
                )
            for doc in payload.get("source_documents", []):
                session.run(
                    """
                    MERGE (c:Company {name: $company})
                    MERGE (d:Document {id: $document_id})
                    SET d.filename = $filename
                    MERGE (c)-[:HAS_DOCUMENT]->(d)
                    """,
                    company=company,
                    document_id=f"{company}:document:{doc.get('document_id', doc.get('filename', 'unknown'))}",
                    filename=doc.get("filename", "unknown"),
                )

    def snapshot(self) -> dict[str, Any]:
        with self.driver.session() as session:
            nodes = session.run(
                """
                MATCH (n)
                RETURN labels(n) AS labels, properties(n) AS props
                LIMIT 100
                """
            ).data()
            edges = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN properties(a) AS source, type(r) AS relation, properties(b) AS target
                LIMIT 200
                """
            ).data()
        return {"backend": "neo4j", "nodes": nodes, "edges": edges}
