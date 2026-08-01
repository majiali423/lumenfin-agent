#!/usr/bin/env python3
"""Deterministic offline validation for runtime and RAG concurrency invariants."""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lumenfin.checkpoint_store import CheckpointConflictError
from lumenfin.database import RagChunk, RagDocument, RagDocumentRepository
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.market_data import DEFAULT_TICKER_MAP
from lumenfin.rag.indexer import DocumentIndexer
from lumenfin.service import LumenFinAnalysisService
from lumenfin.ticker_resolve import set_ticker_directory_for_tests
from tests.test_graph_routing import build_test_config


class OfflineMarketDataClient:
    backend_name = "offline"
    provider = "offline"
    fallback_provider = "offline"

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        return {
            "provider": "offline",
            "symbol": symbol or DEFAULT_TICKER_MAP.get(company, company),
            "company": company,
            "current_price": 100.0,
            "monthly_return": 0.01,
            "market_cap": 1_000_000_000_000,
            "trailing_pe": 25.0,
            "currency": "USD",
            "sector": "Technology",
            "industry": "Technology",
            "fifty_two_week_high": 120.0,
            "fifty_two_week_low": 80.0,
            "status": "ok",
            "from_cache": False,
            "provider_chain": ["offline"],
        }


class FirstCallBarrierLLM(LocalFallbackLLMClient):
    def __init__(self, barrier: Barrier) -> None:
        super().__init__()
        self._barrier = barrier
        self._seen_lock = Lock()
        self._seen_threads: set[int] = set()

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        thread = get_ident()
        with self._seen_lock:
            first = thread not in self._seen_threads
            self._seen_threads.add(thread)
        if first:
            self._barrier.wait(timeout=10)
        return super().chat(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)


class OfflineVectorStore:
    def __init__(
        self,
        *,
        pause_tenants: set[str] | None = None,
        fail_after_write_tenants: set[str] | None = None,
    ) -> None:
        self._lock = Lock()
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.index_calls = 0
        self.pause_tenants = set(pause_tenants or set())
        self.fail_after_write_tenants = set(fail_after_write_tenants or set())
        self.entered = Event()
        self.release = Event()

    def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        *,
        tenant_id: str,
        source_document_id: str,
        content_hash: str = "",
        session_id: str | None = None,
        replace_existing: bool = True,
    ) -> dict[str, int]:
        del content_hash, session_id
        if tenant_id in self.pause_tenants:
            self.entered.set()
            if not self.release.wait(timeout=10):
                raise RuntimeError("offline validator timed out waiting to resume indexing")
        with self._lock:
            self.index_calls += 1
            if replace_existing:
                self.rows = {
                    key: value
                    for key, value in self.rows.items()
                    if key[:2] != (tenant_id, source_document_id)
                }
            for chunk in chunks:
                key = (tenant_id, source_document_id, str(chunk["chunk_id"]))
                self.rows[key] = {
                    **chunk,
                    "tenant_id": tenant_id,
                    "source_document_id": source_document_id,
                }
        if tenant_id in self.fail_after_write_tenants:
            raise RuntimeError(f"injected post-vector failure for {tenant_id}")
        return {"chunks_indexed": len(chunks), "embed_calls": 1}

    def rows_for(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.rows.values() if row["tenant_id"] == tenant_id]

    def delete_by_source_document(self, *, tenant_id: str, source_document_id: str) -> int:
        with self._lock:
            before = len(self.rows)
            self.rows = {
                key: row
                for key, row in self.rows.items()
                if key[:2] != (tenant_id, source_document_id)
            }
            return before - len(self.rows)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.5)))
    return round(ordered[index], 2)


def dispose_service(service: LumenFinAnalysisService) -> None:
    service.repository.engine.dispose()
    service.rag_repository.engine.dispose()
    service.checkpoint_repo.engine.dispose()


def scenario_different_threads(root: Path) -> tuple[int, int, list[float], list[str]]:
    config = replace(build_test_config(root / "different-threads"), rag_enabled=False)
    service = LumenFinAnalysisService(
        config,
        llm_client=LocalFallbackLLMClient(),
        market_data_client=OfflineMarketDataClient(),
    )
    requests = [
        (f"offline-thread-{index}", "Apple" if index % 2 == 0 else "NVIDIA")
        for index in range(10)
    ]

    def run(item: tuple[str, str]) -> tuple[float, str | None]:
        thread_id, company = item
        started = time.perf_counter()
        try:
            response = service.analyze(
                f"Analyze {company} FY2025 profitability and R&D intensity.",
                thread_id=thread_id,
                export_artifacts=False,
            )
            result = response["result"]
            checkpoint = response["checkpoint"]
            assert result["companies"] == [company]
            assert checkpoint["thread_id"] == thread_id
            assert checkpoint["state"]["companies"] == [company]
            telemetry = result["run_telemetry"]
            assert telemetry["total_prompt_tokens"] >= 0
            assert telemetry["total_completion_tokens"] >= 0
            assert all(span["prompt_tokens"] >= 0 for span in telemetry["node_spans"])
            return (time.perf_counter() - started) * 1000, None
        except Exception as exc:  # noqa: BLE001 - validation reports all unexpected errors
            return (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(run, requests))
    latencies = [item[0] for item in outcomes]
    errors = [item[1] for item in outcomes if item[1] is not None]
    dispose_service(service)
    return len(requests), len(requests) - len(errors), latencies, errors


def scenario_same_thread(root: Path) -> tuple[int, int, int, list[float], list[str]]:
    config = replace(build_test_config(root / "same-thread"), rag_enabled=False)
    service = LumenFinAnalysisService(
        config,
        llm_client=FirstCallBarrierLLM(Barrier(2)),
        market_data_client=OfflineMarketDataClient(),
    )
    queries = [
        "Analyze Apple FY2025 profitability and R&D intensity.",
        "Analyze NVIDIA FY2025 profitability and R&D intensity.",
    ]

    def run(query: str) -> tuple[str, float, str | None]:
        started = time.perf_counter()
        try:
            service.analyze(query, thread_id="offline-conflict", export_artifacts=False)
            return "success", (time.perf_counter() - started) * 1000, None
        except CheckpointConflictError as exc:
            return "conflict", (time.perf_counter() - started) * 1000, str(exc)
        except Exception as exc:  # noqa: BLE001
            return "error", (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(run, queries))
    successes = sum(1 for status, _, _ in outcomes if status == "success")
    conflicts = sum(1 for status, _, _ in outcomes if status == "conflict")
    errors = [error for status, _, error in outcomes if status == "error" and error]
    assert successes == 1 and conflicts == 1
    dispose_service(service)
    return len(queries), successes, conflicts, [item[1] for item in outcomes], errors


def scenario_rag(root: Path) -> dict[str, int]:
    rag_root = root / "rag"
    rag_root.mkdir(parents=True, exist_ok=True)
    config = build_test_config(rag_root)
    epoch = [100]
    repo = RagDocumentRepository(
        config.database_url, db_path=config.db_path, epoch_fn=lambda: epoch[0]
    )
    store = OfflineVectorStore(
        pause_tenants={"tenant-dup"},
        fail_after_write_tenants={"tenant-fail"},
    )
    indexer = DocumentIndexer(rag_store=store, repository=repo, lease_seconds=10)
    document = rag_root / "notes.md"
    document.write_text(
        "# Apple FY2025\n\nApple revenue was 100 billion. Supply chain risk remains elevated.",
        encoding="utf-8",
    )

    pending = indexer.enqueue_file(document, tenant_id="tenant-dup")
    with ThreadPoolExecutor(max_workers=1) as pool:
        winning_worker = pool.submit(
            indexer.process_pending,
            pending["document_id"],
            tenant_id="tenant-dup",
        )
        assert store.entered.wait(timeout=10)
        losing_worker = indexer.process_pending(pending["document_id"], tenant_id="tenant-dup")
        in_progress_count = int(losing_worker["status"] == "indexing")
        false_ready_count = int(losing_worker["status"] in {"ready", "skipped_duplicate"})
        assert in_progress_count == 1
        assert false_ready_count == 0
        store.release.set()
        winner = winning_worker.result(timeout=10)
    assert winner["status"] == "ready"
    duplicate = indexer.process_pending(pending["document_id"], tenant_id="tenant-dup")
    assert duplicate["status"] == "skipped_duplicate"
    assert store.index_calls == 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        tenant_receipts = list(
            pool.map(lambda tenant: indexer.index_file(document, tenant_id=tenant), ["tenant-a", "tenant-b"])
        )
    assert tenant_receipts[0]["document_id"] != tenant_receipts[1]["document_id"]
    assert {row["tenant_id"] for row in store.rows_for("tenant-a")} == {"tenant-a"}
    assert {row["tenant_id"] for row in store.rows_for("tenant-b")} == {"tenant-b"}

    failure_document = rag_root / "failure.md"
    failure_document.write_text(
        "# Failure injection\n\nA failed index must not leave vectors or chunks.",
        encoding="utf-8",
    )
    failed_receipt = indexer.index_file(failure_document, tenant_id="tenant-fail")
    assert failed_receipt["status"] == "failed"
    failed_document_id = failed_receipt["document_id"]
    orphan_vector_count = len(store.rows_for("tenant-fail"))
    orphan_chunk_count = len(
        repo.list_chunks(tenant_id="tenant-fail", source_document_ids=[failed_document_id])
    )
    assert orphan_vector_count == 0
    assert orphan_chunk_count == 0

    lease_document = rag_root / "lease-recovery.md"
    lease_document.write_text(
        "# Lease recovery\n\nOnly the current fenced owner may publish this document.",
        encoding="utf-8",
    )
    lease_pending = indexer.enqueue_file(lease_document, tenant_id="tenant-lease")
    claimed_a, owns_a = repo.claim_pending_document(
        document_id=lease_pending["document_id"], tenant_id="tenant-lease",
        index_owner="owner-a", lease_seconds=10,
    )
    assert owns_a
    epoch[0] = 111
    claimed_b, owns_b = repo.claim_pending_document(
        document_id=lease_pending["document_id"], tenant_id="tenant-lease",
        index_owner="owner-b", lease_seconds=10,
    )
    stale_claim_recovered_count = int(owns_b and claimed_b["index_attempt"] == 2)
    stale_finalize = repo.finalize_index_ready(
        document_id=lease_pending["document_id"], tenant_id="tenant-lease",
        index_owner="owner-a", index_attempt=1, filename=lease_pending["filename"],
        content_hash=lease_pending["content_hash"], contexts=[], chunk_count=0,
        source_path=str(lease_document),
    )
    lease_lost_finalize_rejected_count = int(not stale_finalize)
    lease_chunk = {
        "chunk_id": "lease-recovery-chunk", "document_id": "lease-recovery-context",
        "filename": lease_document.name, "page": 1, "text": "new owner lease data",
        "companies": [], "chunk_type": "narrative", "char_count": 20,
    }
    store.index_chunks(
        [lease_chunk], tenant_id="tenant-lease",
        source_document_id=lease_pending["document_id"],
        content_hash=lease_pending["content_hash"], replace_existing=True,
    )
    repo.replace_chunks(
        source_document_id=lease_pending["document_id"], tenant_id="tenant-lease",
        chunks=[lease_chunk], content_hash=lease_pending["content_hash"],
    )
    stale_cleanup = indexer._fail(
        claimed_a, "stale worker failure", index_owner="owner-a", index_attempt=1
    )
    lease_lost_cleanup_rejected_count = int(
        stale_cleanup["error"] == "lease_lost"
        and bool(store.rows_for("tenant-lease"))
        and bool(repo.list_chunks(tenant_id="tenant-lease"))
    )
    finalized_b = repo.finalize_index_ready(
        document_id=lease_pending["document_id"], tenant_id="tenant-lease",
        index_owner="owner-b", index_attempt=2, filename=lease_pending["filename"],
        content_hash=lease_pending["content_hash"], contexts=[], chunk_count=1,
        source_path=str(lease_document),
    )
    final_ready_after_recovery_count = int(finalized_b)
    assert stale_claim_recovered_count == 1
    assert lease_lost_finalize_rejected_count >= 1
    assert lease_lost_cleanup_rejected_count >= 1
    assert final_ready_after_recovery_count == 1

    with Session(repo.engine) as session:
        document_count = int(session.scalar(select(func.count()).select_from(RagDocument)) or 0)
        chunk_count = int(session.scalar(select(func.count()).select_from(RagChunk)) or 0)
        failed = int(
            session.scalar(
                select(func.count()).select_from(RagDocument).where(RagDocument.index_status == "failed")
            )
            or 0
        )
    assert document_count == 5
    assert chunk_count > 0
    assert failed == 1
    repo.engine.dispose()
    return {
        "document_count": document_count,
        "chunk_count": chunk_count,
        "in_progress_count": in_progress_count,
        "false_ready_count": false_ready_count,
        "failed_document_count": failed,
        "orphan_vector_count": orphan_vector_count,
        "orphan_chunk_count": orphan_chunk_count,
        "stale_claim_recovered_count": stale_claim_recovered_count,
        "lease_lost_finalize_rejected_count": lease_lost_finalize_rejected_count,
        "lease_lost_cleanup_rejected_count": lease_lost_cleanup_rejected_count,
        "final_ready_after_recovery_count": final_ready_after_recovery_count,
    }


def main() -> int:
    set_ticker_directory_for_tests([])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lumenfin-concurrency-") as temp_dir:
        root = Path(temp_dir)
        count_a, success_a, latency_a, errors_a = scenario_different_threads(root)
        count_b, success_b, conflicts_b, latency_b, errors_b = scenario_same_thread(root)
        rag_metrics = scenario_rag(root)

    latencies = latency_a + latency_b
    unexpected = errors_a + errors_b
    output = {
        "request_count": count_a + count_b,
        "success_count": success_a + success_b,
        "conflict_count": conflicts_b,
        "unexpected_error_count": len(unexpected),
        "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 0.95),
        **rag_metrics,
        "unexpected_errors": unexpected,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
