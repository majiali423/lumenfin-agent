"""Functional tests for upload-time RAG indexing (Phase 0/1)."""

from __future__ import annotations

import shutil
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.api.app import create_app
from lumenfin.agents import AgentRuntime
from lumenfin.database import RagDocumentRepository
from lumenfin.graph import LumenFinAgentSystem
from lumenfin.knowledge_store import InMemoryKnowledgeStore
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.memory import ReasoningMemory, SessionMemory
from lumenfin.rag.embeddings import DeterministicEmbeddingProvider
from lumenfin.rag.factory import build_hybrid_retriever
from lumenfin.rag.indexer import DocumentIndexer
from lumenfin.rag.milvus_store import MilvusRAGStore
from lumenfin.service import LumenFinAnalysisService
from fastapi.testclient import TestClient
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class CountingEmbedder:
    def __init__(self, dimension: int = 384) -> None:
        self._inner = DeterministicEmbeddingProvider(dimension=dimension)
        self.calls = 0
        self.texts = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.texts += len(texts)
        return self._inner.embed(texts)


def _apple_markdown(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "# Apple Q1 Diligence Notes",
                "",
                "Apple reported revenue of 400 billion and EBITDA margin expansion in FY2025.",
                "Supply chain risk remains elevated due to concentration in Asia manufacturing.",
                "Services growth and installed base monetization continue to support operating leverage.",
            ]
        ),
        encoding="utf-8",
    )
    return path


class RagProductionIndexTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"rag-prod-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = replace(
            build_test_config(self.root),
            rag_index_mode="async_on_upload",
            rag_tenant_id="tenant-a",
        )
        self.milvus_dir = self.root / "milvus"
        self.milvus_dir.mkdir(parents=True, exist_ok=True)
        self.uri = str(self.milvus_dir / "rag.db")
        self.embedder = CountingEmbedder()
        self.store = MilvusRAGStore(self.uri, self.embedder, collection_name="rag_prod")
        self.repo = RagDocumentRepository(self.config.database_url, db_path=self.config.db_path)
        self.indexer = DocumentIndexer(
            rag_store=self.store,
            repository=self.repo,
            tenant_id=self.config.rag_tenant_id,
        )

    def tearDown(self) -> None:
        self.store.close()
        shutil.rmtree(self.root, ignore_errors=True)

    def test_same_bytes_second_index_skips_embed(self) -> None:
        path = _apple_markdown(self.root / "apple_notes.md")
        first = self.indexer.index_file(path)
        self.assertEqual(first["status"], "ready")
        self.assertGreater(first["chunk_count"], 0)
        self.assertEqual(first["embed_calls"], 1)
        self.assertEqual(self.embedder.calls, 1)

        second = self.indexer.index_file(path)
        self.assertEqual(second["status"], "skipped_duplicate")
        self.assertEqual(second["document_id"], first["document_id"])
        self.assertEqual(second["embed_calls"], 0)
        self.assertEqual(self.embedder.calls, 1)
        self.assertEqual(second["chunk_count"], first["chunk_count"])

    def test_keyword_uses_stored_chunks_not_live_rechunk_only(self) -> None:
        path = _apple_markdown(self.root / "apple_notes.md")
        receipt = self.indexer.index_file(path)
        chunks = self.indexer.list_chunks(source_document_ids=[receipt["document_id"]])
        self.assertGreaterEqual(len(chunks), 1)

        retriever = build_hybrid_retriever(
            self.config,
            rag_store=self.store,
            indexer=self.indexer,
        )
        assert retriever is not None
        hits = retriever.retrieve_for_company(
            query="Apple supply chain risk",
            company="Apple",
            session_id="should-not-matter",
            document_contexts=receipt["contexts"],
            tenant_id=self.config.rag_tenant_id,
            source_document_ids=[receipt["document_id"]],
            use_stored_chunks=True,
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(any("supply chain" in hit["text"].lower() for hit in hits))
        self.assertTrue(any(hit.get("citation", "").endswith("#p1") for hit in hits))

    def test_tenant_isolation_on_vector_search(self) -> None:
        path = _apple_markdown(self.root / "apple_notes.md")
        receipt = self.indexer.index_file(path, tenant_id="tenant-a")
        foreign = self.store.vector_search(
            "Apple supply chain risk",
            tenant_id="tenant-b",
            source_document_ids=[receipt["document_id"]],
            companies=["Apple"],
            top_k=5,
        )
        self.assertEqual(foreign, [])
        own = self.store.vector_search(
            "Apple supply chain risk",
            tenant_id="tenant-a",
            source_document_ids=[receipt["document_id"]],
            companies=["Apple"],
            top_k=5,
        )
        self.assertGreaterEqual(len(own), 1)

    def test_async_retrieval_does_not_reindex(self) -> None:
        path = _apple_markdown(self.root / "apple_notes.md")
        receipt = self.indexer.index_file(path)

        def _forbid_index(*_args, **_kwargs):
            self.fail("async_on_upload retrieval must not call index_documents/index_chunks")

        self.store.index_documents = _forbid_index  # type: ignore[method-assign]
        self.store.index_chunks = _forbid_index  # type: ignore[method-assign]

        retriever = build_hybrid_retriever(
            self.config,
            rag_store=self.store,
            indexer=self.indexer,
        )
        runtime = AgentRuntime(
            session_memory=SessionMemory(),
            knowledge_memory=InMemoryKnowledgeStore(),
            reasoning_memory=ReasoningMemory(),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
            hybrid_retriever=retriever,
            rag_enabled=True,
            rag_index_mode="async_on_upload",
            allow_sample_data=True,
            data_mode="demo",
            fetch_live_fundamentals=False,
            fetch_sec_fundamentals=False,
        )
        state = {
            "query": "Assess Apple supply chain risk from the uploaded notes",
            "companies": ["Apple"],
            "target_symbols": {"Apple": "AAPL"},
            "query_plan": {
                "retrieval_query": "Apple supply chain risk",
                "prefer_uploaded_only": False,
                "analysis_dimensions": ["document_evidence", "risk"],
            },
            "document_contexts": receipt["contexts"],
            "rag_document_ids": [receipt["document_id"]],
            "rag_tenant_id": self.config.rag_tenant_id,
            "rag_index_stats": {
                "mode": "async_on_upload",
                "chunks_indexed": receipt["chunk_count"],
                "documents_indexed": 1,
                "search_only": True,
            },
            "thread_id": "thread-async-1",
            "appendix_search_done": False,
            "run_telemetry": {},
            "audit_log": [],
        }
        update = runtime.retrieval(state)  # type: ignore[arg-type]
        self.assertTrue(update.get("rag_index_stats", {}).get("search_only"))
        evidence = update.get("rag_evidence") or {}
        self.assertIn("Apple", evidence)
        self.assertGreaterEqual(len(evidence["Apple"]), 1)

    def test_service_analyze_upload_indexes_once_across_runs(self) -> None:
        config = replace(self.config, milvus_uri=str(self.milvus_dir / "svc.db"), milvus_collection="rag_prod_svc")
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        system = service._system_for("svc-thread")
        assert system.rag_store is not None
        index_calls = {"n": 0}
        real_index = system.rag_store.index_chunks

        def counted_index(*args, **kwargs):
            index_calls["n"] += 1
            return real_index(*args, **kwargs)

        system.rag_store.index_chunks = counted_index  # type: ignore[method-assign]
        system.document_indexer.rag_store = system.rag_store

        path = _apple_markdown(self.root / "apple_svc.md")
        first = service.analyze(
            query="Review Apple notes for supply chain risk",
            thread_id="run-a",
            export_artifacts=False,
            document_paths=[str(path)],
        )
        self.assertIn(first["workflow_status"], {"completed", "incomplete_data", "needs_clarification"})
        first_stats = (first.get("rag_index") or {}).get("stats") or first["result"].get("rag_index_stats") or {}
        self.assertEqual(first_stats.get("mode"), "async_on_upload")
        self.assertEqual(index_calls["n"], 1)

        second = service.analyze(
            query="Second pass on the same Apple notes",
            thread_id="run-b",
            export_artifacts=False,
            document_paths=[str(path)],
        )
        second_stats = (second.get("rag_index") or {}).get("stats") or {}
        receipts = second_stats.get("receipts") or []
        self.assertTrue(receipts)
        self.assertEqual(receipts[0].get("status"), "skipped_duplicate")
        self.assertEqual(index_calls["n"], 1)

    def test_dimension_mismatch_fails_fast(self) -> None:
        uri = str(self.milvus_dir / "dim_mismatch.db")
        first = MilvusRAGStore(uri, DeterministicEmbeddingProvider(dimension=384), collection_name="dim_test")
        first.index_documents(
            [
                {
                    "document_id": "d1",
                    "filename": "a.md",
                    "detected_companies": ["Apple"],
                    "pages": ["Apple revenue grew."],
                }
            ],
            session_id="s1",
        )
        first.close()
        with self.assertRaises(RuntimeError):
            MilvusRAGStore(uri, DeterministicEmbeddingProvider(dimension=1024), collection_name="dim_test")


class SyncOnRunCompatTestCase(unittest.TestCase):
    def test_default_mode_still_indexes_inside_retrieval(self) -> None:
        root = ROOT / "test_artifacts" / f"rag-sync-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            config = build_test_config(root)
            self.assertEqual(config.rag_index_mode, "sync_on_run")
            system = LumenFinAgentSystem(
                llm_client=LocalFallbackLLMClient(),
                app_config=config,
                market_data_client=FakeMarketDataClient(),
            )
            docs = [
                {
                    "document_id": "apple-sync",
                    "filename": "apple.md",
                    "detected_companies": ["Apple"],
                    "source_type": "markdown",
                    "pages": [
                        "Apple revenue 400 billion.",
                        "Supply chain risk remains elevated for Apple.",
                    ],
                    "excerpt": "Apple revenue 400 billion.",
                    "metric_hints": {"revenue": 400.0},
                    "per_company_metric_hints": {"Apple": {"revenue": 400.0}},
                }
            ]
            result = system.run(
                "Assess Apple supply chain risk",
                thread_id="sync-thread",
                document_contexts=docs,
            )
            stats = result.get("rag_index_stats") or {}
            self.assertGreaterEqual(int(stats.get("chunks_indexed") or 0), 1)
            self.assertNotEqual(stats.get("mode"), "async_on_upload")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class DocumentsApiTestCase(unittest.TestCase):
    def test_index_and_status_endpoints(self) -> None:
        root = ROOT / "test_artifacts" / f"rag-api-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        try:
            config = replace(
                build_test_config(root),
                rag_index_mode="async_on_upload",
                rag_tenant_id="api-tenant",
            )
            app = create_app(
                config,
                llm_client=LocalFallbackLLMClient(),
                market_data_client=FakeMarketDataClient(),
            )
            notes = _apple_markdown(root / "api_apple.md")
            with TestClient(app) as client:
                with notes.open("rb") as handle:
                    response = client.post(
                        "/api/v1/documents/index",
                        files={"files": ("api_apple.md", handle, "text/markdown")},
                        data={"tenant_id": "api-tenant"},
                    )
                self.assertEqual(response.status_code, 200, response.text)
                payload = response.json()
                self.assertEqual(payload["tenant_id"], "api-tenant")
                self.assertEqual(len(payload["documents"]), 1)
                doc = payload["documents"][0]
                self.assertIn(doc["status"], {"ready", "skipped_duplicate"})
                self.assertGreater(doc["chunk_count"], 0)

                status = client.get(f"/api/v1/documents/{doc['document_id']}", params={"tenant_id": "api-tenant"})
                self.assertEqual(status.status_code, 200, status.text)
                body = status.json()
                self.assertEqual(body["index_status"], "ready")
                self.assertEqual(body["content_hash"], doc["content_hash"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
