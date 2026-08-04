#!/usr/bin/env python3
"""Consume Redis RAG index jobs with reserve / ACK / retry / reclaim."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.config import AppConfig
from lumenfin.queueing import RedisQueueManager
from lumenfin.service import LumenFinAnalysisService
from lumenfin.stdio import configure_stdio_utf8
from redis.exceptions import RedisError


def _queue_from_config(config: AppConfig) -> RedisQueueManager:
    assert config.redis_url
    return RedisQueueManager(
        config.redis_url,
        config.redis_index_queue_name,
        max_attempts=config.redis_job_max_attempts,
        reclaim_idle_seconds=config.redis_reclaim_idle_seconds,
        retry_backoff_seconds=config.redis_retry_backoff_seconds,
    )


def _should_ack(receipt: dict) -> bool:
    status = str(receipt.get("status") or "")
    return status in {"ready", "skipped_duplicate"}


def _should_delay_retry(receipt: dict) -> bool:
    status = str(receipt.get("status") or "")
    error = str(receipt.get("error") or "")
    if status == "indexing":
        return True
    if error == "lease_lost":
        return True
    return False


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run LumenFin RAG document index worker.")
    parser.add_argument("--once", action="store_true", help="Process one job (or drain pending) then exit.")
    parser.add_argument("--timeout", type=int, default=5, help="Reserve poll timeout seconds.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    service = LumenFinAnalysisService(config)
    if not config.redis_url:
        print("MAS_REDIS_URL is not set; cannot start index worker.")
        return 1

    worker_id = (os.getenv("MAS_WORKER_ID") or f"index-{os.getpid()}").strip()
    queue = _queue_from_config(config)
    try:
        migrated = queue.migrate_legacy_messages()
    except (ConnectionError, OSError, TimeoutError, RedisError) as exc:
        print(f"Legacy queue migrate deferred (redis unavailable): {exc}")
        queue.reset_connection()
        migrated = 0
    if migrated:
        print(f"Migrated {migrated} legacy queue messages into pending")
    print(
        f"Listening on reliable queue={config.redis_index_queue_name} "
        f"worker_id={worker_id} max_attempts={config.redis_job_max_attempts}"
    )

    def _process_one() -> bool:
        queue.reclaim_stale()
        reserved = queue.reserve(timeout_seconds=args.timeout, worker_id=worker_id)
        if reserved is None:
            return False
        payload = reserved.payload
        document_id = str(payload.get("document_id") or "")
        tenant_id = str(payload.get("tenant_id") or config.rag_tenant_id)
        if not document_id:
            queue.retry(reserved.message_id, worker_id, "missing document_id")
            print(f"NACK message_id={reserved.message_id}: missing document_id")
            return True
        try:
            receipt = service.process_document_index(document_id, tenant_id=tenant_id)
        except Exception as exc:  # noqa: BLE001 - convert to retry/dead-letter
            result = queue.retry(reserved.message_id, worker_id, f"exception: {exc}")
            print(
                f"Retry/DLQ message_id={reserved.message_id} action={result.action} "
                f"attempt={result.attempt} error={exc}"
            )
            if result.action == "requeued":
                time.sleep(config.redis_retry_backoff_seconds)
            return True

        print(
            f"Indexed {receipt['document_id']} status={receipt['status']} "
            f"chunks={receipt['chunk_count']} error={receipt.get('error')} "
            f"message_id={reserved.message_id} attempt={reserved.attempt}"
        )
        if _should_ack(receipt):
            acked = queue.ack(reserved.message_id, worker_id)
            print(f"ACK message_id={reserved.message_id} ok={acked}")
            return True

        error = str(receipt.get("error") or receipt.get("status") or "index_incomplete")
        result = queue.retry(reserved.message_id, worker_id, error)
        print(
            f"Retry/DLQ message_id={reserved.message_id} action={result.action} "
            f"attempt={result.attempt} error={error}"
        )
        if result.action == "requeued" and _should_delay_retry(receipt):
            time.sleep(max(config.redis_retry_backoff_seconds, 0.5))
        elif result.action == "requeued":
            time.sleep(config.redis_retry_backoff_seconds)
        return True

    if args.once:
        handled = _process_one()
        if not handled:
            print("No queued job.")
        return 0

    while True:
        try:
            handled = _process_one()
            if not handled:
                time.sleep(0.1)
        except (ConnectionError, OSError, TimeoutError, RedisError) as exc:
            print(f"Worker redis connection error: {exc}")
            queue.reset_connection()
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
