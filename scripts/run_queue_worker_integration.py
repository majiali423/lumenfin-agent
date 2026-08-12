#!/usr/bin/env python3
"""Queue/worker multi-process infrastructure validation entrypoint."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from queue_worker_integration import docker_ops, hooks, milvus_util, redis_util, scenarios
from queue_worker_integration.docker_ops import DockerUnavailable
from queue_worker_integration.settings import OUTPUT_DIR, IntegrationSettings


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for label, args in {
        "docker": ["docker", "version", "--format", "{{.Server.Version}}"],
        "compose": ["docker", "compose", "version", "--short"],
        "python": [sys.executable, "--version"],
    }.items():
        try:
            proc = subprocess.run(args, capture_output=True, text=True, check=False)
            out[label] = (proc.stdout or proc.stderr or "").strip()
        except Exception as exc:  # noqa: BLE001
            out[label] = f"unavailable: {exc}"
    return out


def _image_versions(settings: IntegrationSettings) -> dict[str, str]:
    mapping = {
        "postgres": "postgres",
        "redis": "redis",
        "milvus": "milvus",
    }
    versions: dict[str, str] = {}
    for key, service in mapping.items():
        info = docker_ops.service_inspect(settings, service)
        versions[key] = str(info.get("Image") or info.get("Service") or "unknown")
    return versions


def _write_logs(settings: IntegrationSettings, log_dir: Path) -> None:
    for name in (
        "postgres",
        "redis",
        "milvus",
        "api-a",
        "api-b",
        "index-worker-a",
        "index-worker-b",
    ):
        docker_ops.write_text(log_dir / f"{name}.log", docker_ops.logs(settings, name, tail=400))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run queue/worker multi-process integration suite.")
    parser.add_argument("--keep", action="store_true", help="Keep containers/volumes after run.")
    parser.add_argument("--skip-load", action="store_true", help="Skip P1 limited load scenarios.")
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="Assume infra/APIs already running; only execute scenarios.",
    )
    args = parser.parse_args()

    settings = IntegrationSettings.from_env()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    hooks.HOOK_ROOT.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "phase": "3.2B",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "versions": _versions(),
        "settings": {
            "database_url_host": f"127.0.0.1:{settings.postgres_host_port}/{settings.postgres_db}",
            "redis_url": settings.redis_url,
            "milvus_uri": settings.milvus_uri,
            "redis_index_queue": settings.redis_index_queue,
            "milvus_collection": settings.milvus_collection,
            "lease_seconds": settings.lease_seconds,
        },
        "blocked": None,
        "postgres_migrations": "not_run",
        "different_thread_success": 0,
        "same_thread_success": 0,
        "same_thread_conflict": 0,
        "duplicate_document_count": 0,
        "duplicate_vector_count": 0,
        "stale_claim_recovered": 0,
        "manual_redelivery": False,
        "stale_finalize_rejected": 0,
        "stale_cleanup_rejected": 0,
        "tenant_leakage_count": 0,
        "orphan_chunk_count": 0,
        "orphan_vector_count": 0,
        "dead_letter_count": 0,
        "ack_idempotent": False,
        "redis_reconnect_job_completed": 0,
        "unexpected_error_count": 0,
        "scenario_status": {},
        "artifacts_dir": str(run_dir),
    }

    exit_code = 0
    try:
        if not args.skip_infra:
            print("Starting infrastructure (postgres/redis/milvus)...")
            docker_ops.down_all(settings, volumes=True)
            docker_ops.up_infra(settings)
            # Wait for published ports / health
            deadline = time.monotonic() + 300
            last_err = None
            while time.monotonic() < deadline:
                try:
                    import importlib.util

                    mig_path = ROOT / "scripts" / "run_integration_migrations.py"
                    spec = importlib.util.spec_from_file_location("run_integration_migrations", mig_path)
                    assert spec and spec.loader
                    mig = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mig)
                    mig.wait_for_postgres(settings.database_url, timeout_seconds=5)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    time.sleep(2)
            else:
                raise TimeoutError(f"postgres not reachable: {last_err}")

            print("Running migrations...")
            migration = scenarios.run_migration_gate(settings, run_dir)
            summary["postgres_migrations"] = migration.get("status", "fail")
            summary["scenario_status"]["migrations"] = migration
            if migration.get("status") != "pass":
                raise RuntimeError(f"migration gate failed: {migration.get('errors')}")

            print("Starting API workers...")
            docker_ops.up_apis(settings)
            docker_ops.wait_http_ok(f"{settings.api_a_url}/health", timeout_seconds=240)
            docker_ops.wait_http_ok(f"{settings.api_b_url}/health", timeout_seconds=240)
            print("Starting index workers...")
            docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")
            summary["image_versions"] = _image_versions(settings)
            summary["api_workers"] = {
                "api-a": docker_ops.container_id(settings, "api-a"),
                "api-b": docker_ops.container_id(settings, "api-b"),
            }
            summary["index_workers"] = {
                "index-worker-a": docker_ops.container_id(settings, "index-worker-a"),
                "index-worker-b": docker_ops.container_id(settings, "index-worker-b"),
            }

        # Ensure app processes exist even for --skip-infra re-runs.
        try:
            docker_ops.wait_http_ok(f"{settings.api_a_url}/health", timeout_seconds=15)
            docker_ops.wait_http_ok(f"{settings.api_b_url}/health", timeout_seconds=15)
        except Exception:
            docker_ops.up_apis(settings)
            docker_ops.wait_http_ok(f"{settings.api_a_url}/health", timeout_seconds=180)
            docker_ops.wait_http_ok(f"{settings.api_b_url}/health", timeout_seconds=180)
        docker_ops.up_workers(settings, "index-worker-a", "index-worker-b")

        print("P0-3 checkpoint CAS...")
        cas = scenarios.run_checkpoint_cas(settings, run_dir)
        summary["scenario_status"]["checkpoint_cas"] = cas
        summary["different_thread_success"] = cas.get("different_thread_success", 0)
        summary["same_thread_success"] = cas.get("same_thread_success", 0)
        summary["same_thread_conflict"] = cas.get("same_thread_conflict", 0)
        summary["unexpected_error_count"] += len(cas.get("errors") or [])

        print("P0-4 duplicate Redis jobs...")
        dup = scenarios.run_duplicate_jobs(settings, run_dir)
        summary["scenario_status"]["duplicate_jobs"] = dup
        summary["duplicate_document_count"] = dup.get("duplicate_document_count", 0)
        summary["duplicate_vector_count"] = dup.get("duplicate_vector_count", 0)
        summary["unexpected_error_count"] += len(dup.get("errors") or [])

        print("P0-5 lease recovery after worker kill (auto reclaim, no manual redelivery)...")
        lease = scenarios.run_lease_recovery(settings, run_dir)
        summary["scenario_status"]["lease_recovery"] = lease
        summary["stale_claim_recovered"] = lease.get("stale_claim_recovered", 0)
        summary["manual_redelivery"] = bool(lease.get("manual_redelivery"))
        summary["unexpected_error_count"] += len(lease.get("errors") or [])

        print("P0-6 stale worker fencing...")
        fence = scenarios.run_stale_fencing(settings, run_dir)
        summary["scenario_status"]["stale_fencing"] = fence
        summary["stale_finalize_rejected"] = fence.get("stale_finalize_rejected", 0)
        summary["stale_cleanup_rejected"] = fence.get("stale_cleanup_rejected", 0)
        summary["unexpected_error_count"] += len(fence.get("errors") or [])

        print("P0-7 tenant isolation...")
        tenant = scenarios.run_tenant_isolation(settings, run_dir)
        summary["scenario_status"]["tenant_isolation"] = tenant
        summary["tenant_leakage_count"] = tenant.get("tenant_leakage_count", 0)
        summary["unexpected_error_count"] += len(tenant.get("errors") or [])

        print("P0-8 dead-letter after max attempts...")
        dead = scenarios.run_dead_letter(settings, run_dir)
        summary["scenario_status"]["dead_letter"] = dead
        summary["dead_letter_count"] = int((dead.get("queue_final") or {}).get("dead_letter") or 0)
        summary["unexpected_error_count"] += len(dead.get("errors") or [])

        print("P0-9 ACK idempotency...")
        ack = scenarios.run_ack_idempotency(settings, run_dir)
        summary["scenario_status"]["ack_idempotency"] = ack
        summary["ack_idempotent"] = ack.get("status") == "pass"
        summary["unexpected_error_count"] += len(ack.get("errors") or [])

        print("P1 Redis restart recovery...")
        redis_restart = scenarios.run_redis_restart(settings, run_dir)
        summary["scenario_status"]["redis_restart"] = redis_restart
        summary["redis_reconnect_job_completed"] = int(
            redis_restart.get("job_completed_after_reconnect") or 0
        )
        summary["unexpected_error_count"] += len(redis_restart.get("errors") or [])

        if not args.skip_load:
            print("P1 limited load...")
            load = scenarios.run_limited_load(settings, run_dir)
            summary["scenario_status"]["limited_load"] = load
            summary["orphan_chunk_count"] = load.get("rag", {}).get("orphan_chunks", 0)
            summary["orphan_vector_count"] = load.get("rag", {}).get("orphan_vectors", 0)
            summary["unexpected_error_count"] += len(load.get("errors") or [])
            summary["unexpected_error_count"] += int(load.get("api", {}).get("c10", {}).get("unexpected_error_count") or 0)
            summary["unexpected_error_count"] += int(load.get("api", {}).get("c20", {}).get("unexpected_error_count") or 0)

        summary["redis_queue"] = redis_util.observe_queue(settings.redis_url, settings.redis_index_queue)
        summary["milvus_rows_total"] = milvus_util.count_vectors(
            settings.milvus_uri, settings.milvus_collection
        )
        _write_logs(settings, run_dir)

        failed = [
            name
            for name, payload in summary["scenario_status"].items()
            if isinstance(payload, dict) and payload.get("status") != "pass"
        ]
        summary["status"] = "pass" if not failed and summary["unexpected_error_count"] == 0 else "fail"
        if failed:
            exit_code = 1
            summary["failed_scenarios"] = failed
    except DockerUnavailable as exc:
        summary["status"] = "blocked"
        summary["blocked"] = f"Docker unavailable: {exc}"
        exit_code = 2
        print(summary["blocked"], file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "fail"
        summary["blocked"] = str(exc)
        summary["unexpected_error_count"] += 1
        exit_code = 1
        print(f"Queue/worker integration failed: {exc}", file=sys.stderr)
        try:
            _write_logs(settings, run_dir)
        except Exception:
            pass
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary_path = run_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        latest = OUTPUT_DIR / "summary.json"
        shutil.copyfile(summary_path, latest)
        print(json.dumps({k: summary[k] for k in (
            "status",
            "postgres_migrations",
            "different_thread_success",
            "same_thread_success",
            "same_thread_conflict",
            "duplicate_document_count",
            "duplicate_vector_count",
            "stale_claim_recovered",
            "manual_redelivery",
            "stale_finalize_rejected",
            "stale_cleanup_rejected",
            "tenant_leakage_count",
            "orphan_chunk_count",
            "orphan_vector_count",
            "dead_letter_count",
            "ack_idempotent",
            "redis_reconnect_job_completed",
            "unexpected_error_count",
            "redis_queue",
            "blocked",
            "artifacts_dir",
        ) if k in summary}, indent=2))
        if not args.keep and not args.skip_infra:
            print("Cleaning integration stack...")
            try:
                docker_ops.down_all(settings, volumes=True)
                try:
                    milvus_util.drop_collection(settings.milvus_uri, settings.milvus_collection)
                except Exception:
                    pass
                redis_util.purge_queue(settings.redis_url, settings.redis_index_queue)
            except Exception as exc:  # noqa: BLE001
                print(f"Cleanup warning: {exc}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
