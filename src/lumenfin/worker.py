from __future__ import annotations

import os
import time

from .config import AppConfig
from .queueing import RedisQueueManager
from .service import LumenFinAnalysisService
from redis.exceptions import RedisError


def execute_analysis_job(
    job_id: str,
    query: str,
    thread_id: str,
    export_artifacts: bool = True,
    document_paths: list[str] | None = None,
    output_format: str | None = None,
    *,
    service: LumenFinAnalysisService | None = None,
) -> None:
    if service is None:
        config = AppConfig.from_env()
        service = LumenFinAnalysisService(config)
    service.run_job(
        job_id=job_id,
        query=query,
        thread_id=thread_id,
        export_artifacts=export_artifacts,
        document_paths=document_paths or [],
        output_format=output_format,
    )


def _queue_from_config(config: AppConfig) -> RedisQueueManager:
    assert config.redis_url
    return RedisQueueManager(
        config.redis_url,
        config.redis_queue_name,
        max_attempts=config.redis_job_max_attempts,
        reclaim_idle_seconds=config.redis_reclaim_idle_seconds,
        retry_backoff_seconds=config.redis_retry_backoff_seconds,
    )


def process_reserved_analysis_message(
    *,
    queue: RedisQueueManager,
    service: LumenFinAnalysisService,
    reserved,
    worker_id: str,
    retry_backoff_seconds: float = 0.0,
) -> str:
    """Process one reserved analysis message. Returns action label for tests/logs."""
    payload = reserved.payload
    job_id = str(payload.get("job_id") or "")
    try:
        execute_analysis_job(
            job_id=job_id,
            query=payload["query"],
            thread_id=payload["thread_id"],
            export_artifacts=payload.get("export_artifacts", True),
            document_paths=payload.get("document_paths", []),
            output_format=payload.get("output_format"),
            service=service,
        )
    except Exception as exc:  # noqa: BLE001 - persist failure then retry/DLQ
        if job_id:
            try:
                service.repository.update_job_status(
                    job_id,
                    status="failed",
                    error_message=str(exc),
                )
            except Exception as status_exc:  # noqa: BLE001
                print(f"Failed to persist job failure for {job_id}: {status_exc}")
        result = queue.retry(reserved.message_id, worker_id, f"exception: {exc}")
        print(
            f"Analysis retry/DLQ message_id={reserved.message_id} "
            f"action={result.action} attempt={result.attempt} error={exc}"
        )
        if result.action == "requeued" and retry_backoff_seconds > 0:
            time.sleep(retry_backoff_seconds)
        return result.action
    acked = queue.ack(reserved.message_id, worker_id)
    print(f"Analysis ACK message_id={reserved.message_id} job_id={job_id} ok={acked}")
    return "acked" if acked else "ack_miss"


def work_forever() -> None:
    config = AppConfig.from_env()
    if not config.redis_url:
        raise RuntimeError("MAS_REDIS_URL is required to start the Redis worker.")
    worker_id = (os.getenv("MAS_WORKER_ID") or f"analysis-{os.getpid()}").strip()
    queue = _queue_from_config(config)
    queue.migrate_legacy_messages()
    service = LumenFinAnalysisService(config)
    while True:
        try:
            queue.reclaim_stale()
            reserved = queue.reserve(timeout_seconds=5, worker_id=worker_id)
        except (ConnectionError, OSError, TimeoutError, RedisError) as exc:
            print(f"Analysis worker redis connection error: {exc}")
            queue.reset_connection()
            time.sleep(1.0)
            continue
        if reserved is None:
            continue
        process_reserved_analysis_message(
            queue=queue,
            service=service,
            reserved=reserved,
            worker_id=worker_id,
            retry_backoff_seconds=config.redis_retry_backoff_seconds,
        )
