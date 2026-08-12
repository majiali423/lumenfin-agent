from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.database import RagDocumentRepository
from lumenfin.rag.indexer import DocumentIndexer, canonical_document_id, content_hash_bytes
from lumenfin.rag.milvus_store import MilvusRAGStore
from lumenfin.rag.embeddings import DeterministicEmbeddingProvider

from . import dbutil, docker_ops, hooks, http_client, milvus_util, redis_util
from .settings import IntegrationSettings, OUTPUT_DIR


def _sample_doc(name: str, company: str, body: str | None = None) -> tuple[str, bytes]:
    text = body or (
        f"# {company} FY notes\n\n"
        f"{company} revenue grew with strong enterprise demand.\n"
        f"Gross margin remained durable for {company}.\n"
    )
    return name, text.encode("utf-8")


def _original_filename(filename: str) -> str:
    # save_uploaded_files prefixes "{uuid8}_"
    if len(filename) > 9 and filename[8] == "_":
        return filename[9:]
    return filename


def _docs_by_original_name(body: dict[str, Any]) -> dict[str, str]:
    return {
        _original_filename(str(item["filename"])): str(item["document_id"])
        for item in body.get("documents") or []
    }


def _wait_lease_expired(expires_at: int | None, *, pad_seconds: float = 0.5) -> None:
    """Wait until int(now) > expires_at so claim CAS `expires < now_epoch` can succeed."""
    if expires_at is None:
        time.sleep(pad_seconds)
        return
    target = int(expires_at)
    while int(time.time()) <= target:
        time.sleep(0.05)
    if pad_seconds > 0:
        time.sleep(pad_seconds)


def run_migration_gate(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "fail",
        "empty_db_bootstrap": False,
        "repeat_safe": False,
        "fail_fast_without_002": False,
        "fail_fast_message_ok": False,
        "start_after_002": False,
        "data_preserved": False,
        "errors": [],
    }
    try:
        import importlib.util

        mig_path = ROOT / "scripts" / "run_integration_migrations.py"
        spec = importlib.util.spec_from_file_location("run_integration_migrations", mig_path)
        assert spec and spec.loader
        mig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mig)

        # Host-side migration path against published Postgres port.
        mig.wait_for_postgres(settings.database_url)
        mig.bootstrap_tables(settings.database_url)
        first = mig.apply_sql_files(settings.database_url, mig.MIGRATIONS)
        second = mig.apply_sql_files(settings.database_url, mig.MIGRATIONS)
        result["empty_db_bootstrap"] = True
        result["repeat_safe"] = len(first) == 2 and len(second) == 2
        (log_dir / "migration.json").write_text(
            json.dumps({"first": first, "second": second}, indent=2),
            encoding="utf-8",
        )

        # Seed data then strip lease columns to simulate old schema.
        engine = dbutil.engine_for(settings.database_url)
        with engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO workflow_checkpoints "
                "(thread_id, query, workflow_status, state_json, clarification_questions_json, "
                "last_node, llm_backend, created_at, updated_at, revision) "
                "VALUES ('seed-thread', 'seed', 'completed', '{}', '[]', 'query_planner', "
                "'local-fallback', 't', 't', 1) "
                "ON CONFLICT (thread_id) DO NOTHING"
            )
            conn.exec_driver_sql(
                "INSERT INTO rag_documents "
                "(document_id, tenant_id, filename, content_hash, index_status, error, "
                "indexed_at, chunk_count, contexts_json, source_path, index_owner, "
                "index_lease_expires, index_attempt, created_at, updated_at) "
                "VALUES ('doc-seed', 'tenant-seed', 'seed.md', 'hash-seed', 'ready', NULL, "
                "'t', 1, '[]', NULL, NULL, NULL, 0, 't', 't') "
                "ON CONFLICT (document_id) DO NOTHING"
            )
            conn.exec_driver_sql(
                "INSERT INTO rag_chunks "
                "(chunk_id, document_id, source_document_id, tenant_id, filename, page, text, "
                "companies_json, chunk_type, char_count, content_hash, created_at) "
                "VALUES ('chunk-seed', 'doc-seed', 'doc-seed', 'tenant-seed', 'seed.md', 1, "
                "'seed text', '[]', 'narrative', 9, 'hash-seed', 't') "
                "ON CONFLICT (chunk_id) DO NOTHING"
            )
            conn.exec_driver_sql("ALTER TABLE rag_documents DROP COLUMN IF EXISTS index_owner")
            conn.exec_driver_sql("ALTER TABLE rag_documents DROP COLUMN IF EXISTS index_lease_expires")
            conn.exec_driver_sql("ALTER TABLE rag_documents DROP COLUMN IF EXISTS index_attempt")

        fail_fast_ok = False
        message_ok = False
        try:
            RagDocumentRepository(settings.database_url)
        except RuntimeError as exc:
            fail_fast_ok = True
            message_ok = "002_add_rag_index_lease.sql" in str(exc)
            (log_dir / "migration_fail_fast.txt").write_text(str(exc), encoding="utf-8")
        result["fail_fast_without_002"] = fail_fast_ok
        result["fail_fast_message_ok"] = message_ok

        mig.apply_sql_files(settings.database_url, [mig.MIGRATIONS[1]])
        RagDocumentRepository(settings.database_url)
        result["start_after_002"] = True
        result["data_preserved"] = (
            dbutil.count_checkpoints(settings.database_url, "seed-thread") == 1
            and dbutil.count_documents(settings.database_url, tenant_id="tenant-seed") == 1
            and dbutil.count_chunks(settings.database_url, tenant_id="tenant-seed") == 1
        )
        if all(
            [
                result["empty_db_bootstrap"],
                result["repeat_safe"],
                result["fail_fast_without_002"],
                result["fail_fast_message_ok"],
                result["start_after_002"],
                result["data_preserved"],
            ]
        ):
            result["status"] = "pass"
    except Exception as exc:  # noqa: BLE001
        result["errors"].append(str(exc))
    return result


def run_checkpoint_cas(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "fail",
        "different_thread_success": 0,
        "different_thread_conflict": 0,
        "same_thread_success": 0,
        "same_thread_conflict": 0,
        "api_workers": {},
        "evidence": [],
        "errors": [],
    }
    try:
        health_a = docker_ops.wait_http_ok(f"{settings.api_a_url}/health")
        health_b = docker_ops.wait_http_ok(f"{settings.api_b_url}/health")
        out["api_workers"] = {
            "api-a": {
                "pid": health_a.get("pid"),
                "worker_id": health_a.get("worker_id"),
                "container": docker_ops.container_id(settings, "api-a"),
            },
            "api-b": {
                "pid": health_b.get("pid"),
                "worker_id": health_b.get("worker_id"),
                "container": docker_ops.container_id(settings, "api-b"),
            },
        }
        # In containers both processes often report pid=1; require distinct worker/container identity.
        if health_a.get("worker_id") == health_b.get("worker_id"):
            out["errors"].append("api-a and api-b reported the same worker_id")
        if out["api_workers"]["api-a"]["container"] == out["api_workers"]["api-b"]["container"]:
            out["errors"].append("api-a and api-b resolved to the same container id")

        # Scenario A: different threads
        def _one(i: int) -> dict[str, Any]:
            url = settings.api_a_url if i % 2 == 0 else settings.api_b_url
            thread_id = f"it-diff-{uuid4().hex[:10]}"
            query = "Analyze NVIDIA fundamentals briefly."
            result = http_client.analyze(url, query, thread_id)
            return {
                "thread_id": thread_id,
                "status_code": result.status_code,
                "worker_id": result.headers.get("x-worker-id"),
                "worker_pid": result.headers.get("x-worker-pid"),
                "revision": (result.body or {}).get("checkpoint", {}).get("revision")
                if isinstance(result.body, dict)
                else None,
                "companies": (result.body or {}).get("state", {}).get("companies")
                if isinstance(result.body, dict)
                else None,
                "url": url,
                "elapsed_ms": result.elapsed_ms,
                "body_error": None if result.status_code < 400 else result.body,
            }

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_one, i) for i in range(10)]
            rows = [future.result() for future in as_completed(futures)]
        out["evidence"].extend(rows)
        successes = [row for row in rows if row["status_code"] == 200]
        conflicts = [row for row in rows if row["status_code"] == 409]
        out["different_thread_success"] = len(successes)
        out["different_thread_conflict"] = len(conflicts)
        for row in successes:
            ckpt = dbutil.fetch_checkpoint(settings.database_url, row["thread_id"])
            if ckpt is None or dbutil.count_checkpoints(settings.database_url, row["thread_id"]) != 1:
                out["errors"].append(f"checkpoint missing/duplicated for {row['thread_id']}")

        # Scenario B: same thread concurrent CAS across processes
        hooks.reset_dir(hooks.CHECKPOINT_BARRIER)
        hooks.arm(hooks.CHECKPOINT_BARRIER)
        thread_id = f"it-same-{uuid4().hex[:10]}"
        query_a = "Analyze NVIDIA revenue outlook."
        query_b = "Analyze Apple revenue outlook."

        def _same(url: str, query: str) -> http_client.HttpResult:
            return http_client.analyze(url, query, thread_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_same, settings.api_a_url, query_a)
            fut_b = pool.submit(_same, settings.api_b_url, query_b)
            ready = hooks.wait_for_ready_files(hooks.CHECKPOINT_BARRIER, count=2, timeout_seconds=60)
            hooks.release(hooks.CHECKPOINT_BARRIER)
            res_a = fut_a.result()
            res_b = fut_b.result()
        hooks.disarm(hooks.CHECKPOINT_BARRIER)

        same_rows = [
            {
                "label": "api-a",
                "status_code": res_a.status_code,
                "worker_id": res_a.headers.get("x-worker-id"),
                "worker_pid": res_a.headers.get("x-worker-pid"),
                "revision": (res_a.body or {}).get("checkpoint", {}).get("revision")
                if isinstance(res_a.body, dict)
                else None,
                "companies": (res_a.body or {}).get("state", {}).get("companies")
                if isinstance(res_a.body, dict)
                else None,
                "ready_files": [p.name for p in ready],
            },
            {
                "label": "api-b",
                "status_code": res_b.status_code,
                "worker_id": res_b.headers.get("x-worker-id"),
                "worker_pid": res_b.headers.get("x-worker-pid"),
                "revision": (res_b.body or {}).get("checkpoint", {}).get("revision")
                if isinstance(res_b.body, dict)
                else None,
                "companies": (res_b.body or {}).get("state", {}).get("companies")
                if isinstance(res_b.body, dict)
                else None,
            },
        ]
        out["same_thread_evidence"] = same_rows
        codes = sorted(row["status_code"] for row in same_rows)
        out["same_thread_success"] = sum(1 for code in codes if code == 200)
        out["same_thread_conflict"] = sum(1 for code in codes if code == 409)
        final = dbutil.fetch_checkpoint(settings.database_url, thread_id)
        out["final_checkpoint"] = {
            "revision": None if final is None else final["revision"],
            "query": None if final is None else final["query"],
            "count": dbutil.count_checkpoints(settings.database_url, thread_id),
        }
        winner = next((row for row in same_rows if row["status_code"] == 200), None)
        distinct_workers = {
            str(row.get("worker_id") or "")
            for row in same_rows
            if row.get("worker_id")
        }
        if (
            out["different_thread_success"] == 10
            and out["different_thread_conflict"] == 0
            and out["same_thread_success"] == 1
            and out["same_thread_conflict"] == 1
            and final is not None
            and final["revision"] == 1
            and winner is not None
            and final["query"]
            and dbutil.count_checkpoints(settings.database_url, thread_id) == 1
            and health_a.get("worker_id") != health_b.get("worker_id")
            and out["api_workers"]["api-a"]["container"] != out["api_workers"]["api-b"]["container"]
            and len(distinct_workers) == 2
        ):
            out["status"] = "pass"
        else:
            out["errors"].append("checkpoint CAS assertions failed")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
        hooks.disarm(hooks.CHECKPOINT_BARRIER)
    (log_dir / "checkpoint_cas.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_duplicate_jobs(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "fail",
        "duplicate_document_count": 0,
        "duplicate_vector_count": 0,
        "chunk_count": 0,
        "vector_count": 0,
        "final_status": None,
        "queue_final": None,
        "errors": [],
    }
    try:
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
        content = _sample_doc("dup.md", "NVIDIA", f"NVIDIA duplicate job body {uuid4().hex}\n")[1]
        digest = content_hash_bytes(content)
        tenant = f"tenant-dup-{uuid4().hex[:8]}"
        # Upload once via API to register pending + path
        indexed = http_client.multipart_index(
            f"{settings.api_a_url}/api/v1/documents/index",
            files=[("dup.md", content)],
            tenant_id=tenant,
            async_mode=True,
        )
        if indexed.status_code != 200:
            raise RuntimeError(f"index upload failed: {indexed.body}")
        document_id = indexed.body["documents"][0]["document_id"]
        # Intentionally enqueue duplicates on shared Redis queue.
        redis_util.enqueue_index_job(
            settings.redis_url,
            settings.redis_index_queue,
            document_id=document_id,
            tenant_id=tenant,
            count=3,
        )
        deadline = time.monotonic() + 120
        record = None
        while time.monotonic() < deadline:
            record = dbutil.fetch_document(settings.database_url, document_id, tenant)
            depths = redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
            if (
                record
                and record["index_status"] == "ready"
                and depths.get("pending", 0) == 0
                and depths.get("processing", 0) == 0
                and depths.get("dead_letter", 0) == 0
            ):
                break
            time.sleep(0.5)
        out["final_status"] = None if record is None else record["index_status"]
        out["duplicate_document_count"] = dbutil.count_documents(
            settings.database_url, tenant_id=tenant, content_hash=digest
        )
        record = dbutil.fetch_document(settings.database_url, document_id, tenant)
        out["chunk_count"] = int((record or {}).get("chunk_count") or 0)
        vector_count = 0
        deadline_vectors = time.monotonic() + 30
        while time.monotonic() < deadline_vectors:
            vector_count = milvus_util.count_vectors(
                settings.milvus_uri,
                settings.milvus_collection,
                tenant_id=tenant,
                source_document_id=document_id,
            )
            if out["chunk_count"] > 0 and vector_count == out["chunk_count"]:
                break
            time.sleep(0.5)
        out["vector_count"] = vector_count
        out["duplicate_vector_count"] = max(0, out["vector_count"] - out["chunk_count"])
        out["queue_final"] = redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
        out["workers"] = {
            "index-worker-a": docker_ops.container_id(settings, "index-worker-a"),
            "index-worker-b": docker_ops.container_id(settings, "index-worker-b"),
        }
        if (
            out["duplicate_document_count"] == 1
            and out["final_status"] == "ready"
            and out["chunk_count"] > 0
            and out["vector_count"] == out["chunk_count"]
            and out["duplicate_vector_count"] == 0
            and out["queue_final"]["pending"] == 0
            and out["queue_final"]["processing"] == 0
            and out["queue_final"]["dead_letter"] == 0
        ):
            out["status"] = "pass"
        else:
            out["errors"].append("duplicate job idempotency assertions failed")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
    (log_dir / "duplicate_jobs.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_lease_recovery(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    """Kill worker A after reserve+claim; Worker B must auto-reclaim without manual enqueue."""
    out: dict[str, Any] = {
        "status": "fail",
        "timeline": [],
        "stale_claim_recovered": 0,
        "manual_redelivery": False,
        "queue_final": None,
        "errors": [],
    }
    try:
        redis_util.purge_queue(settings.redis_url, settings.redis_index_queue)
        docker_ops.stop_service(settings, "index-worker-b", kill=True)
        docker_ops.stop_service(settings, "index-worker-a", kill=True)
        hooks.reset_dir(hooks.INDEX_PAUSE)
        hooks.arm(hooks.INDEX_PAUSE)
        docker_ops.up_workers(settings, "index-worker-a")
        deadline_start = time.monotonic() + 60
        while time.monotonic() < deadline_start:
            info = docker_ops.service_inspect(settings, "index-worker-a")
            state = str(info.get("State") or info.get("Status") or "").lower()
            if "running" in state or info.get("ID") or info.get("Container"):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(
                f"index-worker-a failed to start: {docker_ops.logs(settings, 'index-worker-a', tail=80)}"
            )

        content = _sample_doc("crash.md", "Microsoft", f"Microsoft crash recovery {uuid4().hex}\n")[1]
        tenant = f"tenant-crash-{uuid4().hex[:8]}"
        indexed = http_client.multipart_index(
            f"{settings.api_a_url}/api/v1/documents/index",
            files=[("crash.md", content)],
            tenant_id=tenant,
            async_mode=True,
        )
        if indexed.status_code != 200:
            raise RuntimeError(f"upload failed: {indexed.body}")
        document_id = indexed.body["documents"][0]["document_id"]

        claimed_path = hooks.wait_for_claimed(hooks.INDEX_PAUSE, timeout_seconds=90)
        claimed = json.loads(claimed_path.read_text(encoding="utf-8"))
        record = dbutil.fetch_document(settings.database_url, document_id, tenant)
        depths = redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
        out["timeline"].append(
            {
                "event": "worker_a_reserved_and_claimed",
                "ts": time.time(),
                "record": record,
                "claimed": claimed,
                "queue": depths,
            }
        )
        if depths.get("pending", 0) != 0 or depths.get("processing", 0) < 1:
            raise RuntimeError(f"expected message in processing only: {depths}")
        if not record or record["index_status"] != "indexing" or record["index_attempt"] != 1:
            raise RuntimeError(f"unexpected claim state: {record}")
        if not record.get("index_owner"):
            raise RuntimeError("index_owner empty after claim")

        docker_ops.stop_service(settings, "index-worker-a", kill=True)
        out["timeline"].append({"event": "worker_a_killed", "ts": time.time(), "manual_redelivery": False})

        docker_ops.up_workers(settings, "index-worker-b")
        lease_expires = int(record["index_lease_expires"] or 0)
        remaining = max(0.0, lease_expires - time.time() - 1.0)
        if remaining > 0:
            time.sleep(min(remaining, max(0.5, settings.lease_seconds / 2)))
        mid = dbutil.fetch_document(settings.database_url, document_id, tenant)
        out["timeline"].append({"event": "before_expiry_check", "ts": time.time(), "record": mid})
        if mid and mid.get("index_attempt") != 1:
            raise RuntimeError(f"lease stolen before expiry: {mid}")

        # Wait for Redis reclaim idle + DB lease expiry. Do NOT enqueue again.
        _wait_lease_expired(lease_expires, pad_seconds=1.0)
        time.sleep(max(1.0, float(os.getenv("MAS_REDIS_RECLAIM_IDLE_SECONDS", "5"))))
        deadline = time.monotonic() + 120
        recovered = None
        while time.monotonic() < deadline:
            recovered = dbutil.fetch_document(settings.database_url, document_id, tenant)
            depths = redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
            if (
                recovered
                and recovered["index_status"] == "ready"
                and recovered["index_attempt"] == 2
                and depths.get("pending", 0) == 0
                and depths.get("processing", 0) == 0
                and depths.get("dead_letter", 0) == 0
            ):
                break
            time.sleep(0.5)
        out["timeline"].append(
            {
                "event": "after_auto_recovery",
                "ts": time.time(),
                "record": recovered,
                "queue": redis_util.observe_queue(settings.redis_url, settings.redis_index_queue),
            }
        )
        out["final_record"] = recovered
        out["chunk_count"] = int(recovered.get("chunk_count") or 0) if recovered else 0
        vector_count = 0
        deadline_vectors = time.monotonic() + 30
        while time.monotonic() < deadline_vectors:
            vector_count = milvus_util.count_vectors(
                settings.milvus_uri,
                settings.milvus_collection,
                tenant_id=tenant,
                source_document_id=document_id,
            )
            if out["chunk_count"] > 0 and vector_count == out["chunk_count"]:
                break
            time.sleep(0.5)
        out["vector_count"] = vector_count
        out["document_count"] = dbutil.count_documents(
            settings.database_url, tenant_id=tenant, content_hash=content_hash_bytes(content)
        )
        out["queue_final"] = redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
        if (
            recovered
            and recovered["index_status"] == "ready"
            and recovered["index_attempt"] == 2
            and recovered["index_owner"] is None
            and recovered["index_lease_expires"] is None
            and out["document_count"] == 1
            and out["chunk_count"] > 0
            and out["vector_count"] == out["chunk_count"]
            and out["queue_final"]["pending"] == 0
            and out["queue_final"]["processing"] == 0
            and out["queue_final"]["dead_letter"] == 0
            and out["manual_redelivery"] is False
        ):
            out["stale_claim_recovered"] = 1
            out["status"] = "pass"
        else:
            out["errors"].append("lease recovery assertions failed")
        hooks.disarm(hooks.INDEX_PAUSE)
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
        hooks.disarm(hooks.INDEX_PAUSE)
        try:
            docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
        except Exception as restore_exc:  # noqa: BLE001
            out["errors"].append(f"worker restore failed: {restore_exc}")
    (log_dir / "lease_recovery.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_dead_letter(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "fail", "errors": [], "dead_letters": []}
    try:
        from lumenfin.queueing import RedisQueueManager

        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
        redis_util.purge_queue(settings.redis_url, settings.redis_index_queue)
        queue = RedisQueueManager(
            settings.redis_url,
            settings.redis_index_queue,
            max_attempts=3,
            reclaim_idle_seconds=5,
        )
        message_id = queue.enqueue(
            {
                "type": "rag_index",
                "document_id": f"doc-missing-{uuid4().hex[:8]}",
                "tenant_id": f"tenant-dlq-{uuid4().hex[:8]}",
            }
        )
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            depths = queue.depths()
            if depths["dead_letter"] >= 1 and depths["pending"] == 0 and depths["processing"] == 0:
                break
            time.sleep(0.5)
        letters = queue.list_dead_letters(limit=10)
        out["dead_letters"] = letters
        out["queue_final"] = queue.depths()
        out["seed_message_id"] = message_id
        letter = next(
            (item for item in letters if item.get("message_id") == message_id),
            letters[0] if letters else None,
        )
        if (
            letter
            and int(letter.get("attempt") or 0) >= 3
            and letter.get("last_error")
            and letter.get("failed_at")
            and out["queue_final"]["pending"] == 0
            and out["queue_final"]["processing"] == 0
            and out["queue_final"]["dead_letter"] >= 1
        ):
            out["status"] = "pass"
        else:
            out["errors"].append("dead-letter assertions failed")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
    (log_dir / "dead_letter.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_ack_idempotency(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "fail", "errors": []}
    try:
        from lumenfin.queueing import RedisQueueManager

        queue = RedisQueueManager(settings.redis_url, f"{settings.redis_index_queue}-acktest")
        queue.purge()
        message_id = queue.enqueue({"hello": "world"})
        reserved = queue.reserve(timeout_seconds=2, worker_id="ack-tester")
        assert reserved is not None
        first = queue.ack(reserved.message_id, "ack-tester")
        second = queue.ack(reserved.message_id, "ack-tester")
        out.update(
            {
                "message_id": message_id,
                "first_ack": first,
                "second_ack": second,
                "depths": queue.depths(),
            }
        )
        if first is True and second is False and out["depths"]["processing"] == 0:
            out["status"] = "pass"
        else:
            out["errors"].append("ack idempotency assertions failed")
        queue.purge()
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
    (log_dir / "ack_idempotency.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_redis_restart(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "fail",
        "disconnect_count": 1,
        "reconnect_count": 0,
        "job_completed_after_reconnect": 0,
        "errors": [],
    }
    try:
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
        redis_util.purge_queue(settings.redis_url, settings.redis_index_queue)
        # Workers should be idle-waiting, then Redis disappears temporarily.
        docker_ops.stop_service(settings, "redis", kill=True)
        time.sleep(2)
        docker_ops.start_service(settings, "redis")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
                out["reconnect_count"] = 1
                break
            except Exception:
                time.sleep(0.5)
        # Ensure workers are alive after Redis returns (restart:no containers may exit).
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
        content = _sample_doc("redis-restart.md", "Apple", f"Apple redis restart {uuid4().hex}\n")[1]
        tenant = f"tenant-redis-{uuid4().hex[:8]}"
        indexed = http_client.multipart_index(
            f"{settings.api_a_url}/api/v1/documents/index",
            files=[("redis-restart.md", content)],
            tenant_id=tenant,
            async_mode=True,
        )
        if indexed.status_code != 200:
            raise RuntimeError(f"upload failed after redis restart: {indexed.body}")
        document_id = indexed.body["documents"][0]["document_id"]
        deadline = time.monotonic() + 120
        record = None
        while time.monotonic() < deadline:
            record = dbutil.fetch_document(settings.database_url, document_id, tenant)
            if record and record["index_status"] == "ready":
                out["job_completed_after_reconnect"] = 1
                break
            time.sleep(0.5)
        out["final_record"] = record
        if out["reconnect_count"] == 1 and out["job_completed_after_reconnect"] == 1:
            out["status"] = "pass"
        else:
            out["errors"].append("redis restart recovery assertions failed")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
    (log_dir / "redis_restart.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_stale_fencing(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "fail",
        "stale_finalize_rejected": 0,
        "stale_cleanup_rejected": 0,
        "stale_renew_rejected": 0,
        "errors": [],
    }
    try:
        tenant = f"tenant-fence-{uuid4().hex[:8]}"
        content = _sample_doc("fence.md", "AMD", f"AMD fencing body {uuid4().hex}\n")[1]
        digest = content_hash_bytes(content)
        document_id = canonical_document_id(tenant, digest)
        upload_dir = OUTPUT_DIR / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        # Use API registration so source_path exists inside containers; fencing itself is host-side against PG/Milvus.
        indexed = http_client.multipart_index(
            f"{settings.api_a_url}/api/v1/documents/index",
            files=[("fence.md", content)],
            tenant_id=tenant,
            async_mode=True,
        )
        document_id = indexed.body["documents"][0]["document_id"]
        # Wait ready via workers.
        deadline = time.monotonic() + 120
        ready = None
        while time.monotonic() < deadline:
            ready = dbutil.fetch_document(settings.database_url, document_id, tenant)
            if ready and ready["index_status"] == "ready":
                break
            time.sleep(0.5)
        if not ready or ready["index_status"] != "ready":
            raise RuntimeError(f"document not ready for fencing setup: {ready}")

        # Reset to indexing with stale owner/attempt to simulate recovered A after B won.
        # Practical approach: claim as A, expire, claim as B, then stale A operations.
        repo = RagDocumentRepository(settings.database_url)
        # Force pending then claim A
        with dbutil.engine_for(settings.database_url).begin() as conn:
            conn.exec_driver_sql(
                "UPDATE rag_documents SET index_status='pending', index_owner=NULL, "
                "index_lease_expires=NULL, index_attempt=0, error=NULL "
                f"WHERE document_id='{document_id}' AND tenant_id='{tenant}'"
            )
        # Pause workers so host-side fencing claims are not raced by Redis consumers.
        docker_ops.stop_service(settings, "index-worker-a", kill=True)
        docker_ops.stop_service(settings, "index-worker-b", kill=True)
        owner_a = "stale-owner-a"
        record_a, claimed_a = repo.claim_pending_document(
            document_id=document_id,
            tenant_id=tenant,
            index_owner=owner_a,
            lease_seconds=1,
        )
        if not claimed_a:
            raise RuntimeError(f"failed to claim as A: {record_a}")
        attempt_a = int(record_a["index_attempt"])
        _wait_lease_expired(record_a.get("index_lease_expires"), pad_seconds=0.75)
        owner_b = "owner-b"
        record_b, claimed_b = repo.claim_pending_document(
            document_id=document_id,
            tenant_id=tenant,
            index_owner=owner_b,
            lease_seconds=30,
        )
        if not claimed_b:
            raise RuntimeError(f"failed to claim as B: {record_b}")
        attempt_b = int(record_b["index_attempt"])

        renew_a = repo.renew_index_lease(
            document_id=document_id,
            tenant_id=tenant,
            index_owner=owner_a,
            index_attempt=attempt_a,
            lease_seconds=30,
        )
        ready_a = repo.finalize_index_ready(
            document_id=document_id,
            tenant_id=tenant,
            index_owner=owner_a,
            index_attempt=attempt_a,
            filename="fence.md",
            content_hash=digest,
            contexts=[],
            chunk_count=1,
            source_path=None,
        )
        failed_a = repo.finalize_index_failed(
            document_id=document_id,
            tenant_id=tenant,
            index_owner=owner_a,
            index_attempt=attempt_a,
            error="stale",
        )
        # Stale cleanup path via DocumentIndexer._fail renew gate: renew false => cleanup rejected.
        embedder = DeterministicEmbeddingProvider(dimension=384)
        store = MilvusRAGStore(
            settings.milvus_uri,
            embedder,
            collection_name=settings.milvus_collection,
        )
        # Ensure B vectors exist
        store.index_chunks(
            [
                {
                    "chunk_id": f"{document_id}-c0",
                    "document_id": f"{document_id}-c0",
                    "text": "AMD fencing chunk",
                    "filename": "fence.md",
                    "page": 1,
                    "companies": ["AMD"],
                    "chunk_type": "narrative",
                    "char_count": 17,
                }
            ],
            tenant_id=tenant,
            source_document_id=document_id,
            content_hash=digest,
            replace_existing=True,
        )
        before_vectors = milvus_util.count_vectors(
            settings.milvus_uri, settings.milvus_collection, tenant_id=tenant, source_document_id=document_id
        )
        before_chunks = dbutil.count_chunks(settings.database_url, tenant_id=tenant)
        indexer = DocumentIndexer(
            repository=repo,
            rag_store=store,
            tenant_id=tenant,
            lease_seconds=30,
        )
        stale_fail = indexer._fail(  # noqa: SLF001 - intentional fencing probe
            {"document_id": document_id, "tenant_id": tenant, "filename": "fence.md", "content_hash": digest},
            "stale cleanup",
            index_owner=owner_a,
            index_attempt=attempt_a,
        )
        after_vectors = milvus_util.count_vectors(
            settings.milvus_uri, settings.milvus_collection, tenant_id=tenant, source_document_id=document_id
        )
        after_chunks = dbutil.count_chunks(settings.database_url, tenant_id=tenant)
        ready_b = repo.finalize_index_ready(
            document_id=document_id,
            tenant_id=tenant,
            index_owner=owner_b,
            index_attempt=attempt_b,
            filename="fence.md",
            content_hash=digest,
            contexts=[{"filename": "fence.md", "text": "ok"}],
            chunk_count=max(1, after_chunks),
            source_path=None,
        )
        final = dbutil.fetch_document(settings.database_url, document_id, tenant)
        out.update(
            {
                "renew_a": renew_a,
                "ready_a": ready_a,
                "failed_a": failed_a,
                "stale_fail_error": stale_fail.get("error"),
                "ready_b": ready_b,
                "before_vectors": before_vectors,
                "after_vectors": after_vectors,
                "before_chunks": before_chunks,
                "after_chunks": after_chunks,
                "final": final,
            }
        )
        out["stale_renew_rejected"] = 0 if renew_a else 1
        out["stale_finalize_rejected"] = int(not ready_a) + int(not failed_a)
        out["stale_cleanup_rejected"] = 1 if stale_fail.get("error") == "lease_lost" else 0
        if (
            not renew_a
            and not ready_a
            and not failed_a
            and stale_fail.get("error") == "lease_lost"
            and ready_b
            and final
            and final["index_status"] == "ready"
            and after_vectors >= before_vectors
            and after_chunks >= before_chunks
        ):
            out["status"] = "pass"
        else:
            out["errors"].append("fencing assertions failed")
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
    (log_dir / "stale_fencing.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_tenant_isolation(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "fail",
        "tenant_leakage_count": 0,
        "errors": [],
    }
    try:
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
        suffix = uuid4().hex[:8]
        tenant_a = f"tenant-a-{suffix}"
        tenant_b = f"tenant-b-{suffix}"
        shared = _sample_doc("shared.md", "NVIDIA", f"NVIDIA shared tenant isolation body {suffix}.\n")[1]
        other_a = _sample_doc("a-only.md", "Apple", f"Apple only body for {tenant_a}.\n")[1]
        other_b = _sample_doc("b-only.md", "Microsoft", f"Microsoft only body for {tenant_b}.\n")[1]

        def _upload(tenant: str, files: list[tuple[str, bytes]]) -> dict[str, Any]:
            result = http_client.multipart_index(
                f"{settings.api_a_url}/api/v1/documents/index",
                files=files,
                tenant_id=tenant,
                async_mode=True,
            )
            if result.status_code != 200:
                raise RuntimeError(f"{tenant} upload failed: {result.body}")
            return result.body

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_a = pool.submit(_upload, tenant_a, [("shared.md", shared), ("a-only.md", other_a)])
            fut_b = pool.submit(_upload, tenant_b, [("shared.md", shared), ("b-only.md", other_b)])
            body_a = fut_a.result()
            body_b = fut_b.result()

        docs_a = _docs_by_original_name(body_a)
        docs_b = _docs_by_original_name(body_b)
        if "shared.md" not in docs_a or "shared.md" not in docs_b:
            raise RuntimeError(f"missing shared upload mapping: a={docs_a} b={docs_b}")
        if docs_a["shared.md"] == docs_b["shared.md"]:
            raise RuntimeError("same content across tenants produced identical document_id")

        def _wait_ready(tenant: str, document_id: str) -> dict[str, Any]:
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                record = dbutil.fetch_document(settings.database_url, document_id, tenant)
                if record and record["index_status"] in {"ready", "failed"}:
                    return record
                time.sleep(0.5)
            raise TimeoutError(f"timeout waiting for {tenant}/{document_id}")

        ready_a_shared = _wait_ready(tenant_a, docs_a["shared.md"])
        ready_b_shared = _wait_ready(tenant_b, docs_b["shared.md"])
        ready_a_only = _wait_ready(tenant_a, docs_a["a-only.md"])
        ready_b_only = _wait_ready(tenant_b, docs_b["b-only.md"])

        chunks_a = dbutil.count_chunks(settings.database_url, tenant_id=tenant_a)
        chunks_b = dbutil.count_chunks(settings.database_url, tenant_id=tenant_b)
        vectors_a = 0
        vectors_b = 0
        deadline_vectors = time.monotonic() + 45
        while time.monotonic() < deadline_vectors:
            vectors_a = milvus_util.count_vectors(
                settings.milvus_uri, settings.milvus_collection, tenant_id=tenant_a
            )
            vectors_b = milvus_util.count_vectors(
                settings.milvus_uri, settings.milvus_collection, tenant_id=tenant_b
            )
            if vectors_a == chunks_a and vectors_b == chunks_b and chunks_a > 0 and chunks_b > 0:
                break
            time.sleep(0.5)

        store = MilvusRAGStore(
            settings.milvus_uri,
            DeterministicEmbeddingProvider(dimension=384),
            collection_name=settings.milvus_collection,
        )
        hits_a = store.vector_search(
            f"NVIDIA shared tenant isolation {suffix}", top_k=10, tenant_id=tenant_a
        )
        hits_b = store.vector_search(
            f"NVIDIA shared tenant isolation {suffix}", top_k=10, tenant_id=tenant_b
        )
        # Hits filtered by tenant must not include the other tenant's source_document_id.
        leak_a = [hit for hit in hits_a if hit.get("source_document_id") in set(docs_b.values())]
        leak_b = [hit for hit in hits_b if hit.get("source_document_id") in set(docs_a.values())]
        # Cross-tenant source id should not return the other tenant's rows.
        cross = milvus_util.count_vectors(
            settings.milvus_uri,
            settings.milvus_collection,
            tenant_id=tenant_a,
            source_document_id=docs_b["shared.md"],
        )
        # Cleanup one tenant and ensure the other remains.
        store.delete_by_source_document(tenant_id=tenant_a, source_document_id=docs_a["a-only.md"])
        vectors_b_after = milvus_util.count_vectors(
            settings.milvus_uri, settings.milvus_collection, tenant_id=tenant_b
        )
        out.update(
            {
                "docs_a": docs_a,
                "docs_b": docs_b,
                "ready": {
                    "a_shared": ready_a_shared["index_status"],
                    "b_shared": ready_b_shared["index_status"],
                    "a_only": ready_a_only["index_status"],
                    "b_only": ready_b_only["index_status"],
                },
                "chunks_a": chunks_a,
                "chunks_b": chunks_b,
                "vectors_a": vectors_a,
                "vectors_b": vectors_b,
                "vectors_b_after_a_cleanup": vectors_b_after,
                "cross_tenant_source_count": cross,
                "leak_a": len(leak_a),
                "leak_b": len(leak_b),
            }
        )
        out["tenant_leakage_count"] = len(leak_a) + len(leak_b) + cross
        if (
            out["tenant_leakage_count"] == 0
            and chunks_a == vectors_a
            and chunks_b == vectors_b
            and vectors_b_after == vectors_b
            and ready_a_shared["index_status"] == "ready"
            and ready_b_shared["index_status"] == "ready"
        ):
            out["status"] = "pass"
        else:
            out["errors"].append("tenant isolation assertions failed")
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
    (log_dir / "tenant_isolation.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out


def run_limited_load(settings: IntegrationSettings, log_dir: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "fail", "api": {}, "rag": {}, "errors": []}
    try:
        def _api_burst(n: int) -> dict[str, Any]:
            latencies: list[float] = []
            success = 0
            conflict = 0
            unexpected = 0
            started = time.perf_counter()

            def one(i: int) -> http_client.HttpResult:
                url = settings.api_a_url if i % 2 == 0 else settings.api_b_url
                return http_client.analyze(
                    url,
                    "Analyze NVIDIA briefly.",
                    f"load-{n}-{i}-{uuid4().hex[:8]}",
                )

            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(one, i) for i in range(n)]
                for future in as_completed(futures):
                    result = future.result()
                    latencies.append(result.elapsed_ms)
                    if result.status_code == 200:
                        success += 1
                    elif result.status_code == 409:
                        conflict += 1
                    else:
                        unexpected += 1
            duration = time.perf_counter() - started
            latencies_sorted = sorted(latencies)
            return {
                "request_count": n,
                "success_count": success,
                "conflict_count": conflict,
                "unexpected_error_count": unexpected,
                "p50_latency_ms": statistics.median(latencies_sorted) if latencies_sorted else None,
                "p95_latency_ms": latencies_sorted[max(0, int(len(latencies_sorted) * 0.95) - 1)]
                if latencies_sorted
                else None,
                "max_latency_ms": max(latencies_sorted) if latencies_sorted else None,
                "requests_per_second": round(n / duration, 3) if duration else None,
            }

        out["api"] = {"c10": _api_burst(10), "c20": _api_burst(20)}

        jobs_submitted = 0
        for tenant in ("tenant-load-a", "tenant-load-b"):
            for i in range(10):
                unique = _sample_doc(f"u-{tenant}-{i}.md", "NVIDIA", f"unique {tenant} {i} {uuid4().hex}\n")[1]
                http_client.multipart_index(
                    f"{settings.api_a_url}/api/v1/documents/index",
                    files=[(f"u-{i}.md", unique)],
                    tenant_id=tenant,
                    async_mode=True,
                )
                jobs_submitted += 1
            shared = _sample_doc("dup-load.md", "Apple", f"dup body for {tenant}\n")[1]
            for _ in range(10):
                http_client.multipart_index(
                    f"{settings.api_b_url}/api/v1/documents/index",
                    files=[("dup-load.md", shared)],
                    tenant_id=tenant,
                    async_mode=True,
                )
                jobs_submitted += 1

        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if redis_util.queue_depth(settings.redis_url, settings.redis_index_queue) == 0:
                # Allow in-flight workers to finish.
                time.sleep(2)
                if redis_util.queue_depth(settings.redis_url, settings.redis_index_queue) == 0:
                    break
            time.sleep(1)

        ready = 0
        failed = 0
        engine = dbutil.engine_for(settings.database_url)
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session
        from lumenfin.database import RagDocument

        with Session(engine) as session:
            for tenant in ("tenant-load-a", "tenant-load-b"):
                ready += int(
                    session.scalar(
                        select(func.count()).select_from(RagDocument).where(
                            RagDocument.tenant_id == tenant,
                            RagDocument.index_status == "ready",
                        )
                    )
                    or 0
                )
                failed += int(
                    session.scalar(
                        select(func.count()).select_from(RagDocument).where(
                            RagDocument.tenant_id == tenant,
                            RagDocument.index_status == "failed",
                        )
                    )
                    or 0
                )
        chunks = sum(
            dbutil.count_chunks(settings.database_url, tenant_id=t)
            for t in ("tenant-load-a", "tenant-load-b")
        )
        vectors = sum(
            milvus_util.count_vectors(settings.milvus_uri, settings.milvus_collection, tenant_id=t)
            for t in ("tenant-load-a", "tenant-load-b")
        )
        out["rag"] = {
            "jobs_submitted": jobs_submitted,
            "canonical_documents_ready": ready,
            "failed": failed,
            "chunks_persisted": chunks,
            "vectors_persisted": vectors,
            "orphan_chunks": max(0, chunks - vectors),
            "orphan_vectors": max(0, vectors - chunks),
            "queue_depth": redis_util.queue_depth(settings.redis_url, settings.redis_index_queue),
        }
        if (
            out["api"]["c10"]["unexpected_error_count"] == 0
            and out["api"]["c20"]["unexpected_error_count"] == 0
            and out["rag"]["queue_depth"] == 0
            and out["rag"]["orphan_chunks"] == 0
            and out["rag"]["orphan_vectors"] == 0
        ):
            out["status"] = "pass"
        else:
            out["errors"].append("limited load assertions failed")
    except Exception as exc:  # noqa: BLE001
        out["errors"].append(str(exc))
    (log_dir / "limited_load.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return out
