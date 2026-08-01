#!/usr/bin/env python3
"""Deterministic offline validation for runtime and RAG concurrency invariants."""

from __future__ import annotations

import json
import statistics
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock, get_ident
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


class BarrierRepository(RagDocumentRepository):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._barrier = Barrier(2)
        self._lock = Lock()
        self._remaining = 2

    def find_ready_by_hash(self, *, tenant_id: str, content_hash: str):
        record = super().find_ready_by_hash(tenant_id=tenant_id, content_hash=content_hash)
        wait = False
        if record is None:
            with self._lock:
                if self._remaining:
                    self._remaining -= 1
                    wait = True
        if wait:
            self._barrier.wait(timeout=10)
        return record


class OfflineVectorStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self.rows: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.index_calls = 0

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
        return {"chunks_indexed": len(chunks), "embed_calls": 1}

    def rows_for(self, tenant_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self.rows.values() if row["tenant_id"] == tenant_id]


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


def scenario_rag(root: Path) -> tuple[int, int]:
    rag_root = root / "rag"
    rag_root.mkdir(parents=True, exist_ok=True)
    config = build_test_config(rag_root)
    repo = BarrierRepository(config.database_url, db_path=config.db_path)
    store = OfflineVectorStore()
    indexer = DocumentIndexer(rag_store=store, repository=repo)
    document = rag_root / "notes.md"
    document.write_text(
        "# Apple FY2025\n\nApple revenue was 100 billion. Supply chain risk remains elevated.",
        encoding="utf-8",
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        duplicates = list(pool.map(lambda _: indexer.index_file(document, tenant_id="tenant-dup"), range(2)))
    assert len({item["document_id"] for item in duplicates}) == 1
    assert sorted(item["status"] for item in duplicates) == ["ready", "skipped_duplicate"]
    assert store.index_calls == 1

    with ThreadPoolExecutor(max_workers=2) as pool:
        tenant_receipts = list(
            pool.map(lambda tenant: indexer.index_file(document, tenant_id=tenant), ["tenant-a", "tenant-b"])
        )
    assert tenant_receipts[0]["document_id"] != tenant_receipts[1]["document_id"]
    assert {row["tenant_id"] for row in store.rows_for("tenant-a")} == {"tenant-a"}
    assert {row["tenant_id"] for row in store.rows_for("tenant-b")} == {"tenant-b"}

    with Session(repo.engine) as session:
        document_count = int(session.scalar(select(func.count()).select_from(RagDocument)) or 0)
        chunk_count = int(session.scalar(select(func.count()).select_from(RagChunk)) or 0)
        failed = int(
            session.scalar(
                select(func.count()).select_from(RagDocument).where(RagDocument.index_status == "failed")
            )
            or 0
        )
    assert document_count == 3
    assert chunk_count > 0
    assert failed == 0
    repo.engine.dispose()
    return document_count, chunk_count


def main() -> int:
    set_ticker_directory_for_tests([])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="lumenfin-concurrency-") as temp_dir:
        root = Path(temp_dir)
        count_a, success_a, latency_a, errors_a = scenario_different_threads(root)
        count_b, success_b, conflicts_b, latency_b, errors_b = scenario_same_thread(root)
        document_count, chunk_count = scenario_rag(root)

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
        "document_count": document_count,
        "chunk_count": chunk_count,
        "unexpected_errors": unexpected,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if not unexpected else 1


if __name__ == "__main__":
    raise SystemExit(main())
