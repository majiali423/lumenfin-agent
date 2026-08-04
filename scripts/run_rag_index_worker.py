#!/usr/bin/env python3
"""Consume Redis RAG index jobs and complete pending document embeddings."""

from __future__ import annotations

import argparse
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


def main() -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description="Run LumenFin RAG document index worker.")
    parser.add_argument("--once", action="store_true", help="Process one job (or drain pending) then exit.")
    parser.add_argument("--timeout", type=int, default=5, help="Redis BLPOP timeout seconds.")
    args = parser.parse_args()

    config = AppConfig.from_env()
    service = LumenFinAnalysisService(config)
    if not config.redis_url:
        print("MAS_REDIS_URL is not set; cannot start index worker.")
        return 1

    queue = RedisQueueManager(config.redis_url, config.redis_index_queue_name)
    print(f"Listening on queue={config.redis_index_queue_name}")

    def _handle(payload: dict) -> None:
        document_id = str(payload.get("document_id") or "")
        tenant_id = str(payload.get("tenant_id") or config.rag_tenant_id)
        if not document_id:
            print("Skip job without document_id")
            return
        receipt = service.process_document_index(document_id, tenant_id=tenant_id)
        print(
            f"Indexed {receipt['document_id']} status={receipt['status']} "
            f"chunks={receipt['chunk_count']} error={receipt.get('error')}"
        )

    if args.once:
        payload = queue.dequeue(timeout_seconds=args.timeout)
        if payload is None:
            print("No queued job.")
            return 0
        _handle(payload)
        return 0

    while True:
        try:
            payload = queue.dequeue(timeout_seconds=args.timeout)
        except Exception as exc:  # noqa: BLE001
            print(f"Worker dequeue error: {exc}")
            time.sleep(1.0)
            continue
        if payload is None:
            time.sleep(0.1)
            continue
        try:
            _handle(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"Worker error: {exc}")
            time.sleep(1.0)


if __name__ == "__main__":
    raise SystemExit(main())
