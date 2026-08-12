"""Deterministic Analysis Worker at-least-once / redelivery tests.

Covers the Redis reserve 鈫?run_job 鈫?commit 鈫?ACK window without Docker.
Docker multi-worker kill/reclaim remains remaining work (see Phase 3.2B index harness).
"""

from __future__ import annotations

import json
import os
import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep worker tests isolated from developer or production environment settings.
os.environ.setdefault("APP_ENV", "test")

from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.queueing import RedisQueueManager
from lumenfin.service import LumenFinAnalysisService
from lumenfin.worker import process_reserved_analysis_message
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config
from tests.test_redis_queue_resilience import _ListRedis


class _CountingAnalyzeService(LumenFinAnalysisService):
    def __init__(self, *args: Any, fail_times: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.analyze_calls = 0
        self._fail_times = fail_times
        self._failures_seen = 0

    def analyze(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.analyze_calls += 1
        if self._failures_seen < self._fail_times:
            self._failures_seen += 1
            raise RuntimeError(f"injected analysis failure #{self._failures_seen}")
        return {
            "llm_backend": "local-fallback",
            "result": {
                "final_report": f"canonical-result-{self.analyze_calls}",
                "run_id": f"run-{self.analyze_calls}",
            },
            "artifacts": {"report": f"artifact-{self.analyze_calls}.md"},
        }


def _service(root: Path, *, fail_times: int = 0, max_attempts: int = 3) -> _CountingAnalyzeService:
    config = replace(
        build_test_config(root),
        redis_url="redis://fake/0",
        redis_queue_name=f"analysis-{uuid4().hex[:8]}",
        redis_job_max_attempts=max_attempts,
        redis_reclaim_idle_seconds=1,
        redis_retry_backoff_seconds=0.0,
        rag_enabled=False,
    )
    return _CountingAnalyzeService(
        config,
        fail_times=fail_times,
        llm_client=LocalFallbackLLMClient(),
        market_data_client=FakeMarketDataClient(),
    )


class AnalysisWorkerRecoveryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = ROOT / "test_artifacts" / f"analysis-worker-{uuid4().hex[:8]}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.backend = _ListRedis()
        self._redis_patch = patch("lumenfin.queueing.Redis.from_url", return_value=self.backend)
        self._redis_patch.start()

    def tearDown(self) -> None:
        self._redis_patch.stop()

    def _queue(self, service: _CountingAnalyzeService) -> RedisQueueManager:
        assert service.config.redis_url
        return RedisQueueManager(
            service.config.redis_url,
            service.config.redis_queue_name,
            max_attempts=service.config.redis_job_max_attempts,
            reclaim_idle_seconds=service.config.redis_reclaim_idle_seconds,
            retry_backoff_seconds=0.0,
        )

    def _enqueue_job(self, service: _CountingAnalyzeService, queue: RedisQueueManager, job_id: str) -> str:
        service.repository.create_job(job_id, thread_id=f"thread-{job_id}", query="Compare Apple and Microsoft")
        return queue.enqueue(
            {
                "job_id": job_id,
                "query": "Compare Apple and Microsoft",
                "thread_id": f"thread-{job_id}",
                "export_artifacts": False,
            }
        )

    def test_fail_during_execute_then_succeed(self) -> None:
        service = _service(self.root, fail_times=1, max_attempts=3)
        queue = self._queue(service)
        job_id = f"job-{uuid4().hex[:8]}"
        self._enqueue_job(service, queue, job_id)

        reserved = queue.reserve(timeout_seconds=1, worker_id="worker-a")
        assert reserved is not None
        action = process_reserved_analysis_message(
            queue=queue,
            service=service,
            reserved=reserved,
            worker_id="worker-a",
        )
        self.assertEqual(action, "requeued")
        self.assertEqual(queue.depths()["pending"], 1)
        self.assertEqual(queue.depths()["processing"], 0)
        self.assertEqual(queue.depths()["dead_letter"], 0)
        failed = service.get_job(job_id, tenant_id="default")
        assert failed is not None
        self.assertEqual(failed["status"], "failed")
        self.assertIn("injected analysis failure", failed["error_message"] or "")

        reserved = queue.reserve(timeout_seconds=1, worker_id="worker-b")
        assert reserved is not None
        self.assertEqual(reserved.attempt, 2)
        action = process_reserved_analysis_message(
            queue=queue,
            service=service,
            reserved=reserved,
            worker_id="worker-b",
        )
        self.assertEqual(action, "acked")
        depths = queue.depths()
        self.assertEqual(depths["pending"], 0)
        self.assertEqual(depths["processing"], 0)
        self.assertEqual(depths["dead_letter"], 0)
        done = service.get_job(job_id, tenant_id="default")
        assert done is not None
        self.assertEqual(done["status"], "completed")
        self.assertIsNone(done["error_message"])
        self.assertEqual(done["result"]["final_report"], "canonical-result-2")
        self.assertEqual(service.analyze_calls, 2)

    def test_max_retries_enter_dead_letter(self) -> None:
        service = _service(self.root, fail_times=99, max_attempts=2)
        queue = self._queue(service)
        job_id = f"job-{uuid4().hex[:8]}"
        self._enqueue_job(service, queue, job_id)

        for worker in ("worker-a", "worker-b"):
            reserved = queue.reserve(timeout_seconds=1, worker_id=worker)
            assert reserved is not None
            process_reserved_analysis_message(
                queue=queue,
                service=service,
                reserved=reserved,
                worker_id=worker,
            )

        depths = queue.depths()
        self.assertEqual(depths["pending"], 0)
        self.assertEqual(depths["processing"], 0)
        self.assertEqual(depths["dead_letter"], 1)
        job = service.get_job(job_id, tenant_id="default")
        assert job is not None
        self.assertEqual(job["status"], "failed")
        self.assertIn("injected analysis failure", job["error_message"] or "")

    def test_commit_before_ack_redelivery_is_idempotent(self) -> None:
        service = _service(self.root, fail_times=0, max_attempts=3)
        queue = self._queue(service)
        job_id = f"job-{uuid4().hex[:8]}"
        self._enqueue_job(service, queue, job_id)

        reserved = queue.reserve(timeout_seconds=1, worker_id="worker-a")
        assert reserved is not None
        # Simulate: run_job commits successfully, then process dies before ACK.
        service.run_job(
            job_id=job_id,
            query="Compare Apple and Microsoft",
            thread_id=f"thread-{job_id}",
            export_artifacts=False,
        )
        first = service.get_job(job_id, tenant_id="default")
        assert first is not None
        self.assertEqual(first["status"], "completed")
        self.assertEqual(first["result"]["final_report"], "canonical-result-1")
        self.assertEqual(queue.depths()["processing"], 1)

        # Force reclaim of the un-acked message.
        processing = self.backend.lists[queue.processing_key]
        msg = json.loads(processing[0])
        msg["reserved_at"] = int(time.time()) - 30
        processing[0] = json.dumps(msg)
        reclaimed = queue.reclaim_stale()
        self.assertEqual(reclaimed["reclaimed"], 1)

        reserved = queue.reserve(timeout_seconds=1, worker_id="worker-b")
        assert reserved is not None
        action = process_reserved_analysis_message(
            queue=queue,
            service=service,
            reserved=reserved,
            worker_id="worker-b",
        )
        self.assertEqual(action, "acked")
        self.assertEqual(service.analyze_calls, 1)
        second = service.get_job(job_id, tenant_id="default")
        assert second is not None
        self.assertEqual(second["status"], "completed")
        self.assertEqual(second["result"], first["result"])
        self.assertIsNone(second["error_message"])
        depths = queue.depths()
        self.assertEqual(depths["pending"], 0)
        self.assertEqual(depths["processing"], 0)
        self.assertEqual(depths["dead_letter"], 0)

    def test_duplicate_messages_share_one_canonical_result(self) -> None:
        service = _service(self.root, fail_times=0, max_attempts=3)
        queue = self._queue(service)
        job_id = f"job-{uuid4().hex[:8]}"
        self._enqueue_job(service, queue, job_id)
        queue.enqueue(
            {
                "job_id": job_id,
                "query": "Compare Apple and Microsoft",
                "thread_id": f"thread-{job_id}",
                "export_artifacts": False,
            }
        )
        self.assertEqual(queue.depths()["pending"], 2)

        actions: list[str] = []
        for worker in ("worker-a", "worker-b"):
            reserved = queue.reserve(timeout_seconds=1, worker_id=worker)
            assert reserved is not None
            actions.append(
                process_reserved_analysis_message(
                    queue=queue,
                    service=service,
                    reserved=reserved,
                    worker_id=worker,
                )
            )

        self.assertEqual(actions, ["acked", "acked"])
        self.assertEqual(service.analyze_calls, 1)
        job = service.get_job(job_id, tenant_id="default")
        assert job is not None
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["final_report"], "canonical-result-1")
        depths = queue.depths()
        self.assertEqual(depths["pending"], 0)
        self.assertEqual(depths["processing"], 0)
        self.assertEqual(depths["dead_letter"], 0)

    def test_completed_job_is_not_downgraded_to_failed(self) -> None:
        service = _service(self.root)
        job_id = f"job-{uuid4().hex[:8]}"
        service.repository.create_job(job_id, thread_id="t", query="q")
        service.repository.update_job_status(
            job_id,
            status="completed",
            result={"final_report": "keep-me"},
            error_message=None,
        )
        service.repository.update_job_status(
            job_id,
            status="failed",
            error_message="should-not-stick",
        )
        job = service.get_job(job_id, tenant_id="default")
        assert job is not None
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["final_report"], "keep-me")
        self.assertIsNone(job["error_message"])

    def test_completed_redelivery_with_mismatched_query_conflicts(self) -> None:
        from lumenfin.database import JobRedeliveryConflict

        service = _service(self.root)
        job_id = f"job-{uuid4().hex[:8]}"
        service.repository.create_job(job_id, thread_id=f"thread-{job_id}", query="Compare Apple and Microsoft")
        service.run_job(
            job_id=job_id,
            query="Compare Apple and Microsoft",
            thread_id=f"thread-{job_id}",
            export_artifacts=False,
        )
        with self.assertRaises(JobRedeliveryConflict):
            service.run_job(
                job_id=job_id,
                query="Different query that must not reuse completed result",
                thread_id=f"thread-{job_id}",
                export_artifacts=False,
            )
        job = service.get_job(job_id, tenant_id="default")
        assert job is not None
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["result"]["final_report"], "canonical-result-1")
        self.assertEqual(service.analyze_calls, 1)


if __name__ == "__main__":
    unittest.main()
