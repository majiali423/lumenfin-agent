"""Functional tests for upload-time RAG indexing (Phase 0/1)."""

from __future__ import annotations

import shutil
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock
from unittest.mock import Mock, patch
from uuid import uuid4

from sqlalchemy import create_engine, func, inspect, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.api.app import create_app
from lumenfin.agents import AgentRuntime
from lumenfin.database import Base, RagChunk, RagDocument, RagDocumentRepository
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


class _BarrierReadyRepository(RagDocumentRepository):
    def __init__(self, *args, parties: int = 2, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._ready_barrier = Barrier(parties)
        self._ready_lock = Lock()
        self._remaining_empty_checks = parties

    def find_ready_by_hash(self, *, tenant_id: str, content_hash: str):
        record = super().find_ready_by_hash(tenant_id=tenant_id, content_hash=content_hash)
        should_wait = False
        if record is None:
            with self._ready_lock:
                if self._remaining_empty_checks > 0:
                    self._remaining_empty_checks -= 1
                    should_wait = True
        if should_wait:
            self._ready_barrier.wait(timeout=10)
        return record


class _BarrierClaimRepository(RagDocumentRepository):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._claim_barrier = Barrier(2)

    def claim_pending_document(self, *, document_id: str, tenant_id: str):
        self._claim_barrier.wait(timeout=10)
        return super().claim_pending_document(document_id=document_id, tenant_id=tenant_id)


class _RecordingVectorStore:
    def __init__(self, *, fail_tenants: set[str] | None = None) -> None:
        self.fail_tenants = set(fail_tenants or set())
        self._lock = Lock()
        self.index_calls: list[tuple[str, str, tuple[str, ...]]] = []
        self.rows: dict[tuple[str, str, str], dict] = {}

    def index_chunks(
        self,
        chunks,
        *,
        tenant_id: str,
        source_document_id: str,
        content_hash: str = "",
        session_id: str | None = None,
        replace_existing: bool = True,
    ):
        if tenant_id in self.fail_tenants:
            raise RuntimeError(f"injected vector failure for {tenant_id}")
        chunk_ids = tuple(str(chunk["chunk_id"]) for chunk in chunks)
        with self._lock:
            self.index_calls.append((tenant_id, source_document_id, chunk_ids))
            if replace_existing:
                self.rows = {
                    key: value
                    for key, value in self.rows.items()
                    if key[:2] != (tenant_id, source_document_id)
                }
            for chunk in chunks:
                key = (tenant_id, source_document_id, str(chunk["chunk_id"]))
                self.rows[key] = {**chunk, "tenant_id": tenant_id, "source_document_id": source_document_id}
        return {"chunks_indexed": len(chunks), "embed_calls": 1}

    def vector_search(
        self,
        query: str,
        *,
        tenant_id: str | None = None,
        source_document_ids: list[str] | None = None,
        **_kwargs,
    ) -> list[dict]:
        with self._lock:
            rows = list(self.rows.values())
        return [
            dict(row)
            for row in rows
            if (tenant_id is None or row["tenant_id"] == tenant_id)
            and (not source_document_ids or row["source_document_id"] in source_document_ids)
        ]

    def delete_by_source_document(self, *, tenant_id: str, source_document_id: str) -> int:
        with self._lock:
            before = len(self.rows)
            self.rows = {
                key: value
                for key, value in self.rows.items()
                if key[:2] != (tenant_id, source_document_id)
            }
            return before - len(self.rows)

    def close(self) -> None:
        return None


class _PausingVectorStore(_RecordingVectorStore):
    def __init__(self, *, fail_after_release: bool = False) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.fail_after_release = fail_after_release

    def index_chunks(self, chunks, **kwargs):
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise RuntimeError("timed out waiting to resume vector indexing")
        if self.fail_after_release:
            raise RuntimeError("injected paused vector failure")
        return super().index_chunks(chunks, **kwargs)


class _CleanupFailingVectorStore(_RecordingVectorStore):
    def delete_by_source_document(self, *, tenant_id: str, source_document_id: str) -> int:
        raise RuntimeError("injected vector cleanup failure")


class _FailureInjectingRepository(RagDocumentRepository):
    fail_replace = False
    fail_ready = False
    fail_delete = False

    def replace_chunks(self, **kwargs) -> None:
        if self.fail_replace:
            raise RuntimeError("injected chunk persistence failure")
        return super().replace_chunks(**kwargs)

    def upsert_document(self, **kwargs):
        if self.fail_ready and kwargs.get("index_status") == "ready":
            raise RuntimeError("injected ready update failure")
        return super().upsert_document(**kwargs)

    def delete_chunks(self, **kwargs) -> int:
        if self.fail_delete:
            raise RuntimeError("injected chunk cleanup failure")
        return super().delete_chunks(**kwargs)


def _repository_counts(repo: RagDocumentRepository) -> tuple[int, int, int]:
    with Session(repo.engine) as session:
        documents = int(session.scalar(select(func.count()).select_from(RagDocument)) or 0)
        chunks = int(session.scalar(select(func.count()).select_from(RagChunk)) or 0)
        failed = int(
            session.scalar(
                select(func.count()).select_from(RagDocument).where(RagDocument.index_status == "failed")
            )
            or 0
        )
    return documents, chunks, failed


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


def _nvidia_markdown(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "# NVIDIA FY2025 Diligence Notes",
                "",
                "NVIDIA reported revenue of 130 billion in FY2025.",
                "Data center demand remains strong while supply constraints remain a risk.",
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


class RagConcurrencyHardeningTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"rag-concurrency-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = replace(build_test_config(self.root), rag_index_mode="async_on_upload")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _indexer(self, store: _RecordingVectorStore) -> tuple[DocumentIndexer, _BarrierReadyRepository]:
        repo = _BarrierReadyRepository(self.config.database_url, db_path=self.config.db_path)
        return DocumentIndexer(rag_store=store, repository=repo), repo

    def test_concurrent_same_tenant_same_content_has_one_canonical_index(self) -> None:
        path = _apple_markdown(self.root / "duplicate.md")
        store = _RecordingVectorStore()
        indexer, repo = self._indexer(store)

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(pool.map(lambda _: indexer.index_file(path, tenant_id="tenant-a"), range(2)))

        self.assertEqual({receipt["document_id"] for receipt in receipts}, {receipts[0]["document_id"]})
        self.assertEqual(sorted(receipt["status"] for receipt in receipts), ["ready", "skipped_duplicate"])
        documents, chunks, failed = _repository_counts(repo)
        self.assertEqual(documents, 1)
        self.assertEqual(chunks, receipts[0]["chunk_count"] or receipts[1]["chunk_count"])
        self.assertEqual(failed, 0)
        self.assertEqual(len(store.index_calls), 1)
        vector_chunk_ids = store.index_calls[0][2]
        self.assertEqual(len(vector_chunk_ids), len(set(vector_chunk_ids)))

        later = indexer.index_file(path, tenant_id="tenant-a")
        self.assertEqual(later["status"], "skipped_duplicate")
        self.assertEqual(later["document_id"], receipts[0]["document_id"])
        self.assertEqual(len(store.index_calls), 1)

    def test_duplicate_pending_workers_write_chunks_and_vectors_once(self) -> None:
        path = _apple_markdown(self.root / "duplicate-workers.md")
        store = _RecordingVectorStore()
        repo = _BarrierClaimRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)
        pending = indexer.enqueue_file(path, tenant_id="tenant-a")
        self.assertEqual(pending["status"], "pending")

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipts = list(
                pool.map(
                    lambda _: indexer.process_pending(pending["document_id"], tenant_id="tenant-a"),
                    range(2),
                )
            )

        self.assertEqual(sum(receipt["status"] == "ready" for receipt in receipts), 1)
        self.assertIn(
            next(receipt["status"] for receipt in receipts if receipt["status"] != "ready"),
            {"indexing", "skipped_duplicate"},
        )
        self.assertEqual(len(store.index_calls), 1)
        documents, chunks, failed = _repository_counts(repo)
        self.assertEqual(documents, 1)
        self.assertGreater(chunks, 0)
        self.assertEqual(failed, 0)

    def test_losing_worker_and_api_report_indexing_until_winner_is_ready(self) -> None:
        path = _apple_markdown(self.root / "paused-winner.md")
        store = _PausingVectorStore()
        repo = RagDocumentRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)
        pending = indexer.enqueue_file(path, tenant_id="tenant-a")
        app = create_app(
            replace(self.config, rag_tenant_id="tenant-a"),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )

        with ThreadPoolExecutor(max_workers=1) as pool:
            winner = pool.submit(indexer.process_pending, pending["document_id"], tenant_id="tenant-a")
            self.assertTrue(store.entered.wait(timeout=10))

            loser = indexer.process_pending(pending["document_id"], tenant_id="tenant-a")
            self.assertEqual(loser["status"], "indexing")
            with TestClient(app) as client:
                current = client.get(
                    f"/api/v1/documents/{pending['document_id']}", params={"tenant_id": "tenant-a"}
                )
                process = client.post(
                    f"/api/v1/documents/{pending['document_id']}/process", params={"tenant_id": "tenant-a"}
                )
            self.assertEqual(current.status_code, 200, current.text)
            self.assertEqual(current.json()["index_status"], "indexing")
            self.assertEqual(process.status_code, 200, process.text)
            self.assertEqual(process.json()["index_status"], "indexing")

            store.release.set()
            completed = winner.result(timeout=10)

        self.assertEqual(completed["status"], "ready")
        self.assertEqual(repo.get_document(pending["document_id"], tenant_id="tenant-a")["index_status"], "ready")

    def test_losing_worker_never_reports_ready_when_paused_winner_fails(self) -> None:
        path = _apple_markdown(self.root / "paused-failure.md")
        store = _PausingVectorStore(fail_after_release=True)
        repo = RagDocumentRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)
        pending = indexer.enqueue_file(path, tenant_id="tenant-a")

        with ThreadPoolExecutor(max_workers=1) as pool:
            winner = pool.submit(indexer.process_pending, pending["document_id"], tenant_id="tenant-a")
            self.assertTrue(store.entered.wait(timeout=10))
            loser = indexer.process_pending(pending["document_id"], tenant_id="tenant-a")
            self.assertEqual(loser["status"], "indexing")
            store.release.set()
            failed = winner.result(timeout=10)

        self.assertEqual(failed["status"], "failed")
        persisted = repo.get_document(pending["document_id"], tenant_id="tenant-a")
        self.assertEqual(persisted["index_status"], "failed")
        self.assertNotEqual(loser["status"], "ready")
        self.assertNotEqual(loser["status"], "skipped_duplicate")

    def test_ready_duplicate_is_the_only_skipped_duplicate_process_result(self) -> None:
        path = _apple_markdown(self.root / "ready-duplicate.md")
        store = _RecordingVectorStore()
        repo = RagDocumentRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)

        ready = indexer.index_file(path, tenant_id="tenant-a")
        duplicate = indexer.process_pending(ready["document_id"], tenant_id="tenant-a")

        self.assertEqual(ready["status"], "ready")
        self.assertEqual(duplicate["status"], "skipped_duplicate")
        self.assertEqual(duplicate["chunk_count"], ready["chunk_count"])
        self.assertEqual(len(store.index_calls), 1)
        self.assertEqual(
            len(repo.list_chunks(tenant_id="tenant-a", source_document_ids=[ready["document_id"]])),
            ready["chunk_count"],
        )

    def test_vector_write_is_compensated_when_chunk_persistence_fails(self) -> None:
        store = _RecordingVectorStore()
        repo = _FailureInjectingRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)
        other = indexer.index_file(_nvidia_markdown(self.root / "other.md"), tenant_id="tenant-b")
        other_vectors = store.vector_search("NVIDIA", tenant_id="tenant-b")
        other_chunks = repo.list_chunks(tenant_id="tenant-b")
        repo.fail_replace = True

        failed = indexer.index_file(_apple_markdown(self.root / "target.md"), tenant_id="tenant-a")

        self.assertEqual(failed["status"], "failed")
        self.assertIn("injected chunk persistence failure", failed["error"] or "")
        self.assertEqual(store.vector_search("Apple", tenant_id="tenant-a"), [])
        self.assertEqual(repo.list_chunks(tenant_id="tenant-a"), [])
        self.assertEqual(store.vector_search("NVIDIA", tenant_id="tenant-b"), other_vectors)
        self.assertEqual(repo.list_chunks(tenant_id="tenant-b"), other_chunks)
        self.assertEqual(repo.get_document(failed["document_id"], tenant_id="tenant-a")["index_status"], "failed")
        self.assertEqual(repo.get_document(other["document_id"], tenant_id="tenant-b")["index_status"], "ready")

    def test_vectors_and_chunks_are_compensated_when_ready_update_fails(self) -> None:
        store = _RecordingVectorStore()
        repo = _FailureInjectingRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)
        repo.fail_ready = True

        failed = indexer.index_file(_apple_markdown(self.root / "ready-update.md"), tenant_id="tenant-a")

        self.assertEqual(failed["status"], "failed")
        self.assertIn("injected ready update failure", failed["error"] or "")
        self.assertEqual(store.vector_search("Apple", tenant_id="tenant-a"), [])
        self.assertEqual(repo.list_chunks(tenant_id="tenant-a"), [])
        persisted = repo.get_document(failed["document_id"], tenant_id="tenant-a")
        self.assertEqual(persisted["index_status"], "failed")
        self.assertEqual(persisted["chunk_count"], 0)

    def test_cleanup_failure_preserves_original_index_error(self) -> None:
        store = _CleanupFailingVectorStore()
        repo = _FailureInjectingRepository(self.config.database_url, db_path=self.config.db_path)
        repo.fail_replace = True
        repo.fail_delete = True
        indexer = DocumentIndexer(rag_store=store, repository=repo)

        failed = indexer.index_file(_apple_markdown(self.root / "cleanup-failure.md"), tenant_id="tenant-a")

        self.assertEqual(failed["status"], "failed")
        self.assertIn("injected chunk persistence failure", failed["error"] or "")
        self.assertIn("cleanup", failed["error"] or "")
        self.assertIn("injected vector cleanup failure", failed["error"] or "")
        self.assertIn("injected chunk cleanup failure", failed["error"] or "")
        persisted = repo.get_document(failed["document_id"], tenant_id="tenant-a")
        self.assertEqual(persisted["index_status"], "failed")
        self.assertEqual(persisted["error"], failed["error"])

    def test_same_content_concurrent_tenants_remain_separate(self) -> None:
        path = _apple_markdown(self.root / "shared.md")
        store = _RecordingVectorStore()
        indexer, repo = self._indexer(store)

        def index_and_retrieve(tenant: str) -> tuple[dict, list[dict]]:
            receipt = indexer.index_file(path, tenant_id=tenant)
            hits = store.vector_search(
                "Apple revenue",
                tenant_id=tenant,
                source_document_ids=[receipt["document_id"]],
            )
            return receipt, hits

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = dict(zip(["tenant-a", "tenant-b"], pool.map(index_and_retrieve, ["tenant-a", "tenant-b"]), strict=True))

        receipt_a, hits_a = outcomes["tenant-a"]
        receipt_b, hits_b = outcomes["tenant-b"]
        self.assertNotEqual(receipt_a["document_id"], receipt_b["document_id"])
        self.assertTrue(hits_a and hits_b)
        self.assertEqual({hit["tenant_id"] for hit in hits_a}, {"tenant-a"})
        self.assertEqual({hit["tenant_id"] for hit in hits_b}, {"tenant-b"})
        self.assertEqual(
            store.vector_search("Apple", tenant_id="tenant-a", source_document_ids=[receipt_b["document_id"]]),
            [],
        )
        self.assertEqual(
            store.vector_search("Apple", tenant_id="tenant-b", source_document_ids=[receipt_a["document_id"]]),
            [],
        )
        documents, chunks, failed = _repository_counts(repo)
        self.assertEqual(documents, 2)
        self.assertEqual(chunks, receipt_a["chunk_count"] + receipt_b["chunk_count"])
        self.assertEqual(failed, 0)
        self.assertEqual({row["tenant_id"] for row in repo.list_chunks(tenant_id="tenant-a")}, {"tenant-a"})
        self.assertEqual({row["tenant_id"] for row in repo.list_chunks(tenant_id="tenant-b")}, {"tenant-b"})

    def test_concurrent_tenant_failure_does_not_pollute_success(self) -> None:
        path_a = _apple_markdown(self.root / "fail-a.md")
        path_b = _apple_markdown(self.root / "ok-b.md")
        store = _RecordingVectorStore(fail_tenants={"tenant-a"})
        indexer, repo = self._indexer(store)

        with ThreadPoolExecutor(max_workers=2) as pool:
            receipt_a, receipt_b = list(
                pool.map(
                    lambda item: indexer.index_file(item[1], tenant_id=item[0]),
                    [("tenant-a", path_a), ("tenant-b", path_b)],
                )
            )

        self.assertEqual(receipt_a["status"], "failed")
        self.assertEqual(receipt_b["status"], "ready")
        self.assertEqual(store.vector_search("Apple", tenant_id="tenant-a"), [])
        hits_b = store.vector_search("Apple", tenant_id="tenant-b")
        self.assertTrue(hits_b)
        self.assertEqual({hit["tenant_id"] for hit in hits_b}, {"tenant-b"})
        self.assertEqual(repo.list_chunks(tenant_id="tenant-a"), [])
        self.assertTrue(repo.list_chunks(tenant_id="tenant-b"))

    def test_different_content_tenants_match_serial_baseline(self) -> None:
        paths = {
            "tenant-a": _apple_markdown(self.root / "apple-a.md"),
            "tenant-b": _nvidia_markdown(self.root / "nvidia-b.md"),
        }
        concurrent_store = _RecordingVectorStore()
        concurrent_repo = RagDocumentRepository(self.config.database_url, db_path=self.config.db_path)
        concurrent_indexer = DocumentIndexer(rag_store=concurrent_store, repository=concurrent_repo)

        def index_and_retrieve(tenant: str) -> tuple[dict, set[str]]:
            receipt = concurrent_indexer.index_file(paths[tenant], tenant_id=tenant)
            hits = concurrent_store.vector_search("financial risk", tenant_id=tenant)
            return receipt, {str(hit["text"]) for hit in hits}

        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent = dict(
                zip(paths, pool.map(index_and_retrieve, paths), strict=True)
            )

        serial_root = self.root / "serial-baseline"
        serial_config = build_test_config(serial_root)
        serial_repo = RagDocumentRepository(serial_config.database_url, db_path=serial_config.db_path)
        serial_store = _RecordingVectorStore()
        serial_indexer = DocumentIndexer(rag_store=serial_store, repository=serial_repo)
        serial: dict[str, tuple[dict, set[str]]] = {}
        for tenant, path in paths.items():
            receipt = serial_indexer.index_file(path, tenant_id=tenant)
            hits = serial_store.vector_search("financial risk", tenant_id=tenant)
            serial[tenant] = (receipt, {str(hit["text"]) for hit in hits})

        for tenant in paths:
            concurrent_receipt, concurrent_texts = concurrent[tenant]
            serial_receipt, serial_texts = serial[tenant]
            self.assertEqual(concurrent_receipt["status"], "ready")
            self.assertEqual(concurrent_receipt["document_id"], serial_receipt["document_id"])
            self.assertEqual(concurrent_receipt["chunk_count"], serial_receipt["chunk_count"])
            self.assertEqual(concurrent_texts, serial_texts)
            self.assertTrue(concurrent_texts)
        self.assertEqual(
            concurrent_store.vector_search(
                "NVIDIA",
                tenant_id="tenant-a",
                source_document_ids=[concurrent["tenant-b"][0]["document_id"]],
            ),
            [],
        )

    def test_persistence_rejects_cross_tenant_document_and_chunk_mutation(self) -> None:
        path = _apple_markdown(self.root / "tenant-boundary.md")
        store = _RecordingVectorStore()
        repo = RagDocumentRepository(self.config.database_url, db_path=self.config.db_path)
        indexer = DocumentIndexer(rag_store=store, repository=repo)
        receipt = indexer.index_file(path, tenant_id="tenant-a")
        chunks = repo.list_chunks(tenant_id="tenant-a", source_document_ids=[receipt["document_id"]])
        self.assertTrue(chunks)

        with self.assertRaisesRegex(ValueError, "tenant"):
            repo.upsert_document(
                document_id=receipt["document_id"],
                tenant_id="tenant-b",
                filename=receipt["filename"],
                content_hash=receipt["content_hash"],
                index_status="ready",
                contexts=receipt["contexts"],
                chunk_count=receipt["chunk_count"],
                source_path=str(path),
            )
        with self.assertRaisesRegex(ValueError, "tenant"):
            repo.replace_chunks(
                source_document_id=receipt["document_id"],
                tenant_id="tenant-b",
                chunks=chunks,
                content_hash=receipt["content_hash"],
            )

        stored = repo.get_document(receipt["document_id"], tenant_id="tenant-a")
        self.assertIsNotNone(stored)
        self.assertIsNone(repo.get_document(receipt["document_id"], tenant_id="tenant-b"))


class RagIndexLeaseTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"rag-lease-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.config = build_test_config(self.root)
        self.epoch = 100
        self.repo = RagDocumentRepository(
            self.config.database_url, db_path=self.config.db_path, epoch_fn=lambda: self.epoch
        )
        self.path = _apple_markdown(self.root / "lease.md")
        self.store = _RecordingVectorStore()
        self.indexer = DocumentIndexer(
            rag_store=self.store, repository=self.repo, lease_seconds=10
        )

    def tearDown(self) -> None:
        self.repo.engine.dispose()
        shutil.rmtree(self.root, ignore_errors=True)

    def _pending(self) -> dict:
        receipt = self.indexer.enqueue_file(self.path, tenant_id="tenant-a")
        self.assertEqual(receipt["status"], "pending")
        return receipt

    def _claim(self, document_id: str, owner: str) -> tuple[dict, bool]:
        return self.repo.claim_pending_document(
            document_id=document_id,
            tenant_id="tenant-a",
            index_owner=owner,
            lease_seconds=10,
        )

    def _finalize_ready(self, pending: dict, owner: str, attempt: int) -> bool:
        return self.repo.finalize_index_ready(
            document_id=pending["document_id"],
            tenant_id="tenant-a",
            index_owner=owner,
            index_attempt=attempt,
            filename=pending["filename"],
            content_hash=pending["content_hash"],
            contexts=[],
            chunk_count=0,
            source_path=str(self.path),
        )

    def test_active_lease_cannot_be_stolen_and_only_owner_can_finalize(self) -> None:
        pending = self._pending()
        claimed_a, owns_a = self._claim(pending["document_id"], "owner-a")
        self.assertTrue(owns_a)
        self.assertEqual(claimed_a["index_attempt"], 1)
        self.epoch = 105

        observed, owns_b = self._claim(pending["document_id"], "owner-b")

        self.assertFalse(owns_b)
        self.assertEqual(observed["index_status"], "indexing")
        self.assertEqual(observed["index_owner"], "owner-a")
        self.assertEqual(observed["index_attempt"], 1)
        self.assertTrue(self._finalize_ready(pending, "owner-a", 1))

    def test_expired_lease_is_reclaimed_and_stale_finalize_is_rejected(self) -> None:
        pending = self._pending()
        _, owns_a = self._claim(pending["document_id"], "owner-a")
        self.assertTrue(owns_a)
        self.epoch = 111
        claimed_b, owns_b = self._claim(pending["document_id"], "owner-b")

        self.assertTrue(owns_b)
        self.assertEqual(claimed_b["index_owner"], "owner-b")
        self.assertEqual(claimed_b["index_attempt"], 2)
        self.assertFalse(self._finalize_ready(pending, "owner-a", 1))
        self.assertFalse(
            self.repo.finalize_index_failed(
                document_id=pending["document_id"], tenant_id="tenant-a",
                index_owner="owner-a", index_attempt=1, error="stale failure",
            )
        )
        stored = self.repo.get_document(pending["document_id"], tenant_id="tenant-a")
        self.assertEqual((stored["index_owner"], stored["index_attempt"]), ("owner-b", 2))

    def test_stale_worker_cannot_cleanup_new_owner_data(self) -> None:
        pending = self._pending()
        claimed_a, _ = self._claim(pending["document_id"], "owner-a")
        self.epoch = 111
        _, owns_b = self._claim(pending["document_id"], "owner-b")
        self.assertTrue(owns_b)
        chunk = {
            "chunk_id": "lease-chunk", "document_id": "lease-context",
            "filename": "lease.md", "page": 1, "text": "new owner data",
            "companies": ["Apple"], "chunk_type": "narrative", "char_count": 14,
        }
        self.store.index_chunks(
            [chunk], tenant_id="tenant-a", source_document_id=pending["document_id"],
            content_hash=pending["content_hash"], replace_existing=True,
        )
        self.repo.replace_chunks(
            source_document_id=pending["document_id"], tenant_id="tenant-a",
            chunks=[chunk], content_hash=pending["content_hash"],
        )

        stale = self.indexer._fail(
            claimed_a, "old worker failure", index_owner="owner-a", index_attempt=1
        )

        self.assertEqual(stale["error"], "lease_lost")
        self.assertTrue(self.store.vector_search("new", tenant_id="tenant-a"))
        self.assertTrue(self.repo.list_chunks(tenant_id="tenant-a"))
        self.assertEqual(
            self.repo.get_document(pending["document_id"], tenant_id="tenant-a")["index_owner"],
            "owner-b",
        )
        self.assertTrue(self._finalize_ready(pending, "owner-b", 2))

    def test_stale_indexing_is_recovered_to_one_ready_document(self) -> None:
        pending = self._pending()
        self._claim(pending["document_id"], "abandoned")
        self.epoch = 111

        recovered = self.indexer.process_pending(pending["document_id"], tenant_id="tenant-a")

        self.assertEqual(recovered["status"], "ready")
        stored = self.repo.get_document(pending["document_id"], tenant_id="tenant-a")
        self.assertEqual(stored["index_status"], "ready")
        self.assertIsNone(stored["index_owner"])
        self.assertIsNone(stored["index_lease_expires"])
        self.assertEqual(stored["index_attempt"], 2)
        documents, chunks, failed = _repository_counts(self.repo)
        self.assertEqual((documents, failed), (1, 0))
        self.assertEqual(chunks, recovered["chunk_count"])

    def test_sqlite_old_rag_table_gets_lease_columns(self) -> None:
        legacy_path = self.root / "legacy.db"
        engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "CREATE TABLE rag_documents (document_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, "
                "filename TEXT NOT NULL, content_hash TEXT NOT NULL, index_status TEXT NOT NULL, "
                "error TEXT, indexed_at TEXT, chunk_count INTEGER NOT NULL DEFAULT 0, "
                "contexts_json TEXT NOT NULL DEFAULT '[]', source_path TEXT, "
                "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
            )
        engine.dispose()

        migrated = RagDocumentRepository(f"sqlite:///{legacy_path.as_posix()}", db_path=legacy_path)
        columns = {column["name"] for column in inspect(migrated.engine).get_columns("rag_documents")}
        self.assertTrue({"index_owner", "index_lease_expires", "index_attempt"}.issubset(columns))
        migrated.engine.dispose()

    def test_postgresql_lease_migration_and_fail_fast_message(self) -> None:
        migration = ROOT / "migrations" / "postgresql" / "002_add_rag_index_lease.sql"
        sql = migration.read_text(encoding="utf-8")
        self.assertEqual(sql.upper().count("ADD COLUMN IF NOT EXISTS"), 3)
        schema = Mock()
        schema.has_table.return_value = True
        schema.get_columns.return_value = [{"name": "document_id"}, {"name": "index_status"}]
        with patch.object(Base.metadata, "create_all"), patch(
            "lumenfin.database.inspect", return_value=schema
        ):
            with self.assertRaisesRegex(
                RuntimeError, r"rag_documents.*002_add_rag_index_lease\.sql.*psql"
            ):
                RagDocumentRepository("postgresql+psycopg://user:password@localhost/lumenfin")


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
