from __future__ import annotations

import os
import signal
import threading
import time

from .config import AppConfig
from .llm import shutdown_llm_http_clients
from .provider_resilience import redact_provider_message
from .provider_resilience import close_shared_http_clients
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
    tenant_id: str | None = None,
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
        tenant_id=tenant_id,
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
            tenant_id=payload.get("tenant_id"),
            service=service,
        )
    except Exception as exc:  # noqa: BLE001 - persist failure then retry/DLQ
        safe_error = redact_provider_message(str(exc))
        if job_id:
            try:
                service.repository.update_job_status(
                    job_id,
                    status="failed",
                    error_message=safe_error,
                )
            except Exception as status_exc:  # noqa: BLE001
                print(
                    f"Failed to persist job failure for {job_id}: "
                    f"{redact_provider_message(str(status_exc))}"
                )
        result = queue.retry(reserved.message_id, worker_id, f"exception: {safe_error}")
        print(
            f"Analysis retry/DLQ message_id={reserved.message_id} "
            f"action={result.action} attempt={result.attempt} "
            f"error_type={type(exc).__name__} error={safe_error}"
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
    stop_event = threading.Event()

    def _request_stop(signum, _frame) -> None:
        print(f"Analysis worker shutdown requested signal={signum}", flush=True)
        stop_event.set()

    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    try:
        while not stop_event.is_set():
            try:
                queue.reclaim_stale()
                reserved = queue.reserve(timeout_seconds=1, worker_id=worker_id)
            except (ConnectionError, OSError, TimeoutError, RedisError) as exc:
                print(
                    "Analysis worker redis connection error: "
                    f"{redact_provider_message(str(exc))}"
                )
                queue.reset_connection()
                stop_event.wait(1.0)
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
    finally:
        rag_store = getattr(service, "_rag_store", None)
        if rag_store is not None:
            rag_store.close()
        shutdown_llm_http_clients()
        close_shared_http_clients()
        print("Analysis worker shutdown complete", flush=True)
