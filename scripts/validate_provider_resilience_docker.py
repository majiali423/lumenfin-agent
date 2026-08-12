#!/usr/bin/env python3
"""Docker dual-API provider resilience Scenario G (real containers; no OS-process fallback)."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from queue_worker_integration import docker_ops
from queue_worker_integration.docker_ops import DockerUnavailable
from queue_worker_integration.settings import ENV_FILE, IntegrationSettings
from queue_worker_integration import scenarios as queue_scenarios

COMPOSE_BASE = ROOT / "docker-compose.integration.yml"
COMPOSE_OVERLAY = ROOT / "docker-compose.provider-resilience.yml"
OUTPUT_ROOT = ROOT / "outputs" / "provider_resilience"

UNCLOSED_RE = re.compile(r"unclosed\s+(client|transport)", re.I)
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
PG_ERR_RE = re.compile(r"(pool|psycopg|OperationalError|Too many connections)", re.I)
REDIS_ERR_RE = re.compile(r"(redis\.(exceptions|RedisError)|ConnectionError.*redis)", re.I)
MILVUS_ERR_RE = re.compile(r"(MilvusException|pymilvus|gRPC.*milvus)", re.I)


def _compose_cmd(settings: IntegrationSettings, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_BASE),
        "-f",
        str(COMPOSE_OVERLAY),
        "--env-file",
        str(ENV_FILE),
        "-p",
        settings.project,
        *args,
    ]


def _run_compose(
    settings: IntegrationSettings,
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    docker_ops._require_docker()  # noqa: SLF001
    return subprocess.run(
        _compose_cmd(settings, *args),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def _inspect_container(cid: str) -> dict[str, Any]:
    proc = subprocess.run(
        ["docker", "inspect", cid],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout or "").strip()[:500]}
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"error": "inspect json decode failed"}
    if not data:
        return {}
    item = data[0]
    cfg = item.get("Config") or {}
    state = item.get("State") or {}
    return {
        "id": str(item.get("Id") or "")[:12],
        "id_full": str(item.get("Id") or ""),
        "name": str(item.get("Name") or "").lstrip("/"),
        "hostname": cfg.get("Hostname"),
        "pid": state.get("Pid"),
        "status": state.get("Status"),
        "restart_count": int(item.get("RestartCount") or 0),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
    }


def _fault_plan() -> list[tuple[str, str]]:
    scenarios = (
        ["success"] * 10
        + ["503_then_success"] * 4
        + ["429_then_success"] * 3
        + ["always_503"] * 2
        + ["permanent_400"] * 1
    )
    assert len(scenarios) == 20
    return [("api-a" if i < 10 else "api-b", s) for i, s in enumerate(scenarios)]


def _save_diagnostics(settings: IntegrationSettings, docker_dir: Path) -> None:
    docker_dir.mkdir(parents=True, exist_ok=True)
    ps = _run_compose(settings, "ps", check=False)
    (docker_dir / "compose-ps.txt").write_text(
        (ps.stdout or "") + (ps.stderr or ""), encoding="utf-8"
    )
    for name in ("api-a", "api-b", "provider-stub", "postgres", "redis", "milvus"):
        log_text = _run_compose(settings, "logs", "--no-color", "--tail", "400", name, check=False)
        (docker_dir / f"{name}.log").write_text(
            (log_text.stdout or "") + (log_text.stderr or ""), encoding="utf-8"
        )


def _count_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text or ""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Provider resilience Docker dual-API Scenario G")
    parser.add_argument(
        "--mode",
        choices=("docker",),
        default="docker",
        help="Must be docker. OS-process fallback is intentionally unsupported.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep compose stack after run (default: tear down).",
    )
    parser.add_argument(
        "--skip-infra",
        action="store_true",
        help="Assume stack already running (still requires Docker containers).",
    )
    args, _unknown = parser.parse_known_args()
    if args.mode != "docker":
        print("ERROR: this runner only supports --mode docker", file=sys.stderr)
        return 2

    settings = IntegrationSettings.from_env()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_ROOT / f"docker_{stamp}"
    docker_dir = run_dir / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "phase": "3.3A",
        "scenario": "G_dual_api_docker",
        "mode": "docker",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "compose_files": [str(COMPOSE_BASE.name), str(COMPOSE_OVERLAY.name)],
        "test_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "note": "per-process bulkhead ≠ cross-process global rate limit",
        "status": "fail",
        "errors": [],
    }
    exit_code = 1
    api_a_url = settings.api_a_url
    api_b_url = settings.api_b_url
    stub_url = f"http://127.0.0.1:{int(__import__('os').getenv('PROVIDER_STUB_PORT', '18090'))}"

    try:
        docker_ops._require_docker()  # noqa: SLF001
    except DockerUnavailable as exc:
        summary["status"] = "fail"
        summary["blocked"] = f"Docker unavailable: {exc}"
        summary["errors"] = [summary["blocked"]]
        (docker_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print("FAIL: Docker required; refusing OS-process fallback.", file=sys.stderr)
        return 2

    try:
        if not args.skip_infra:
            print("1/11 down previous stack...")
            _run_compose(settings, "down", "--remove-orphans", "-v", check=False)

            print("2/11 build + start infra (postgres/redis/milvus)...")
            _run_compose(
                settings,
                "up",
                "-d",
                "--build",
                "postgres",
                "redis",
                "etcd",
                "minio",
                "milvus",
            )

            print("3/11 migrations...")
            # Wait for postgres then run host-side migration gate (same as queue/worker integration).
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

            migration = queue_scenarios.run_migration_gate(settings, run_dir)
            summary["postgres_migrations"] = migration.get("status")
            if migration.get("status") != "pass":
                raise RuntimeError(f"migration failed: {migration.get('errors')}")

            print("4/11 start provider-stub...")
            _run_compose(settings, "up", "-d", "--build", "provider-stub")
            docker_ops.wait_http_ok(f"{stub_url}/health", timeout_seconds=120)

            print("5-6/11 start api-a and api-b...")
            _run_compose(settings, "up", "-d", "--build", "api-a", "api-b")
            docker_ops.wait_http_ok(f"{api_a_url}/health", timeout_seconds=240)
            docker_ops.wait_http_ok(f"{api_b_url}/health", timeout_seconds=240)

        print("7/11 health + identities...")
        # Resolve container IDs via compose ps + docker inspect (authoritative).
        identities: dict[str, Any] = {}
        for name in ("api-a", "api-b", "provider-stub"):
            info = _run_compose(settings, "ps", "--format", "json", name, check=False)
            lines = [ln for ln in (info.stdout or "").splitlines() if ln.strip()]
            cid = ""
            if lines:
                try:
                    parsed = json.loads(lines[0] if len(lines) == 1 else f"[{','.join(lines)}]")
                    row = parsed[0] if isinstance(parsed, list) else parsed
                    cid = str(row.get("ID") or row.get("Container") or "")
                except json.JSONDecodeError:
                    cid = ""
            if not cid:
                # Fallback: docker compose ps -q
                q = _run_compose(settings, "ps", "-q", name, check=False)
                cid = (q.stdout or "").strip().splitlines()[0] if (q.stdout or "").strip() else ""
            inspected = _inspect_container(cid) if cid else {}
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                if name.startswith("api-"):
                    body = client.get(f"{api_a_url if name == 'api-a' else api_b_url}/api/v1/provider-resilience/identity").json()
                else:
                    body = client.get(f"{stub_url}/health").json()
            identities[name] = {
                "compose_id": cid[:12] if cid else None,
                "inspect": inspected,
                "endpoint": body if name.startswith("api-") else body,
            }

        (docker_dir / "container-identities.json").write_text(
            json.dumps(identities, indent=2), encoding="utf-8"
        )
        summary["container_identities"] = {
            name: {
                "container_id": (identities[name].get("inspect") or {}).get("id"),
                "hostname": (identities[name].get("inspect") or {}).get("hostname")
                or (identities[name].get("endpoint") or {}).get("hostname"),
                "pid": (identities[name].get("endpoint") or {}).get("pid")
                or (identities[name].get("inspect") or {}).get("pid"),
                "worker_id": (identities[name].get("endpoint") or {}).get("worker_id"),
                "restart_count_before": (identities[name].get("inspect") or {}).get("restart_count"),
            }
            for name in ("api-a", "api-b")
        }

        a_id = summary["container_identities"]["api-a"]["container_id"]
        b_id = summary["container_identities"]["api-b"]["container_id"]
        a_host = summary["container_identities"]["api-a"]["hostname"]
        b_host = summary["container_identities"]["api-b"]["hostname"]
        if not a_id or not b_id or a_id == b_id:
            raise RuntimeError(f"API containers must be distinct: a={a_id} b={b_id}")
        if not a_host or not b_host or a_host == b_host:
            raise RuntimeError(f"API hostnames must differ: a={a_host} b={b_host}")

        # Reset stub counters.
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            client.get(f"{stub_url}/__stub__/reset")

        print("8/11 execute Scenario G (10→A, 10→B)...")
        plan = _fault_plan()
        bases = {"api-a": api_a_url, "api-b": api_b_url}
        rows: list[dict[str, Any]] = []
        latencies: list[float] = []
        traces: list[dict[str, Any]] = []
        request_ids: list[str] = []
        logical_ids: list[str] = []
        client_ids: dict[str, set[str]] = {"api-a": set(), "api-b": set()}
        max_inflight: dict[str, int] = {"api-a": 0, "api-b": 0}
        configured_limit = 4

        def one(api: str, scenario: str) -> dict[str, Any]:
            started = time.perf_counter()
            with httpx.Client(timeout=45.0, trust_env=False) as client:
                resp = client.post(
                    f"{bases[api]}/api/v1/provider-resilience/probe",
                    json={
                        "scenario": scenario,
                        "prompt": f"{api}:{scenario}:{uuid4().hex[:8]}",
                        "max_attempts": 2 if scenario == "always_503" else 3,
                    },
                )
            elapsed = (time.perf_counter() - started) * 1000.0
            body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            return {
                "api": api,
                "scenario": scenario,
                "http_status": resp.status_code,
                "latency_ms": elapsed,
                "body": body,
            }

        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(one, api, scenario) for api, scenario in plan]
            for fut in as_completed(futures):
                rows.append(fut.result())

        success = degraded = expected_failure = unexpected_failure = 0
        fallback = deadline_exceeded = provider_busy = 0
        api_counts = {"api-a": 0, "api-b": 0}
        context_leakage = 0
        logical_to_request: dict[str, str] = {}
        duplicate_logical = 0

        for row in rows:
            api = row["api"]
            api_counts[api] += 1
            latencies.append(row["latency_ms"])
            body = row.get("body") or {}
            rid = str(body.get("request_id") or "")
            request_ids.append(rid)
            if body.get("worker_id") and body.get("worker_id") != api:
                context_leakage += 1
            if body.get("hostname") and body.get("hostname") not in {
                summary["container_identities"][api]["hostname"],
                api,
            }:
                # Hostname should match container hostname (api-a / api-b).
                if body.get("hostname") != summary["container_identities"][api]["hostname"]:
                    context_leakage += 1
            cid = body.get("client_instance_id") or body.get("client_id")
            if cid:
                client_ids[api].add(str(cid))
            max_inflight[api] = max(
                max_inflight[api], int(body.get("llm_max_inflight_seen") or 0)
            )
            configured_limit = int(body.get("llm_max_inflight_configured") or configured_limit)
            for event in body.get("trace") or []:
                traces.append(event)
                lid = str(event.get("logical_call_id") or "")
                if lid:
                    logical_ids.append(lid)
                    prior = logical_to_request.get(lid)
                    if prior is not None and prior != rid:
                        duplicate_logical += 1
                    logical_to_request[lid] = rid
                    if event.get("request_id") and event.get("request_id") != rid:
                        context_leakage += 1
            scenario = row["scenario"]
            if row["http_status"] != 200:
                unexpected_failure += 1
                continue
            if scenario == "permanent_400":
                if body.get("ok"):
                    unexpected_failure += 1
                else:
                    expected_failure += 1
                continue
            if body.get("degraded") or body.get("fallback"):
                degraded += 1
                fallback += 1
            elif body.get("ok"):
                success += 1
            else:
                unexpected_failure += 1
            if body.get("error_class") == "deadline_exceeded":
                deadline_exceeded += 1
            if body.get("error_class") == "provider_busy":
                provider_busy += 1

        from lumenfin.provider_resilience import summarize_provider_trace

        trace_summary = summarize_provider_trace(traces)
        with httpx.Client(timeout=10.0, trust_env=False) as client:
            stub = client.get(f"{stub_url}/__stub__/stats").json()

        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted) if latencies_sorted else 0
        p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))] if latencies_sorted else 0

        # Final identity / inflight after traffic.
        post_identities = {}
        for name, url in (("api-a", api_a_url), ("api-b", api_b_url)):
            with httpx.Client(timeout=10.0, trust_env=False) as client:
                post_identities[name] = client.get(f"{url}/api/v1/provider-resilience/identity").json()
            max_inflight[name] = max(
                max_inflight[name],
                int(post_identities[name].get("llm_max_inflight_seen") or 0),
            )

        # Restart counts after scenario (before graceful stop).
        for name in ("api-a", "api-b"):
            cid = summary["container_identities"][name]["container_id"]
            # Re-resolve full id via compose.
            q = _run_compose(settings, "ps", "-q", name, check=False)
            full = (q.stdout or "").strip().splitlines()[0] if (q.stdout or "").strip() else ""
            inspected = _inspect_container(full or cid)
            before = int(summary["container_identities"][name].get("restart_count_before") or 0)
            after = int(inspected.get("restart_count") or 0)
            summary["container_identities"][name]["restart_count_after"] = after
            summary["container_identities"][name]["unexpected_restart_count"] = max(0, after - before)

        print("9/11 graceful stop APIs for lifespan shutdown evidence...")
        _run_compose(settings, "stop", "api-a", "api-b", check=False)
        time.sleep(2)
        api_a_log = _run_compose(settings, "logs", "--no-color", "--tail", "200", "api-a", check=False)
        api_b_log = _run_compose(settings, "logs", "--no-color", "--tail", "200", "api-b", check=False)
        api_a_text = (api_a_log.stdout or "") + (api_a_log.stderr or "")
        api_b_text = (api_b_log.stdout or "") + (api_b_log.stderr or "")
        (docker_dir / "api-a.log").write_text(api_a_text, encoding="utf-8")
        (docker_dir / "api-b.log").write_text(api_b_text, encoding="utf-8")

        shutdown_a = "shared HTTP clients closed" in api_a_text or "provider transport cleanup completed" in api_a_text
        shutdown_b = "shared HTTP clients closed" in api_b_text or "provider transport cleanup completed" in api_b_text
        unclosed = _count_matches(api_a_text, UNCLOSED_RE) + _count_matches(api_b_text, UNCLOSED_RE)
        # Count only shutdown-window tracebacks near the end if possible; use full log conservatively.
        shutdown_tracebacks = 0
        for text in (api_a_text, api_b_text):
            if "provider transport cleanup" in text and TRACEBACK_RE.search(text.split("provider transport cleanup")[-1]):
                shutdown_tracebacks += 1
            elif TRACEBACK_RE.search(text) and "lifespan" in text.lower():
                shutdown_tracebacks += 1

        stub_log = _run_compose(settings, "logs", "--no-color", "--tail", "400", "provider-stub", check=False)
        (docker_dir / "provider-stub.log").write_text(
            (stub_log.stdout or "") + (stub_log.stderr or ""), encoding="utf-8"
        )
        with (docker_dir / "provider-calls.jsonl").open("w", encoding="utf-8") as fh:
            for item in stub.get("requests") or []:
                fh.write(json.dumps(item) + "\n")

        # Dependency error scan (tail logs).
        dep_logs = {}
        for name in ("postgres", "redis", "milvus"):
            lg = _run_compose(settings, "logs", "--no-color", "--tail", "100", name, check=False)
            dep_logs[name] = (lg.stdout or "") + (lg.stderr or "")
            (docker_dir / f"{name}.log").write_text(dep_logs[name], encoding="utf-8")

        combined_api = api_a_text + "\n" + api_b_text
        postgres_errors = _count_matches(combined_api, PG_ERR_RE)
        redis_errors = _count_matches(combined_api, REDIS_ERR_RE)
        milvus_errors = _count_matches(combined_api, MILVUS_ERR_RE)

        ps = _run_compose(settings, "ps", check=False)
        (docker_dir / "compose-ps.txt").write_text(
            (ps.stdout or "") + (ps.stderr or ""), encoding="utf-8"
        )

        cross_share = client_ids["api-a"] & client_ids["api-b"]
        summary.update(
            {
                "api_container_count": 2,
                "api_a": summary["container_identities"]["api-a"],
                "api_b": summary["container_identities"]["api-b"],
                "request_count": len(rows),
                "api_a_count": api_counts["api-a"],
                "api_b_count": api_counts["api-b"],
                "logical_provider_calls": trace_summary.get("logical_provider_calls"),
                "physical_provider_attempts": trace_summary.get("physical_provider_attempts"),
                "retry_amplification_ratio": trace_summary.get("retry_amplification_ratio"),
                "success_count": success,
                "degraded_count": degraded,
                "expected_failure_count": expected_failure,
                "unexpected_failure_count": unexpected_failure,
                "fallback_count": fallback,
                "deadline_exceeded_count": deadline_exceeded,
                "provider_busy_count": provider_busy,
                "provider_context_leakage_count": context_leakage,
                "duplicate_logical_call_id_count": duplicate_logical,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "max_latency_ms": round(max(latencies_sorted), 2) if latencies_sorted else 0,
                "stub_request_count": stub.get("request_count"),
                "client_instance_ids": {k: sorted(v) for k, v in client_ids.items()},
                "cross_process_shared_client_ids": sorted(cross_share),
                "http_client_reuse": {
                    "api_a_unique_instance_ids": len(client_ids["api-a"]),
                    "api_b_unique_instance_ids": len(client_ids["api-b"]),
                    "same_process_reuse_ok": len(client_ids["api-a"]) == 1 and len(client_ids["api-b"]) == 1,
                },
                "bulkhead": {
                    "configured_limit": configured_limit,
                    "api_a_max_inflight": max_inflight["api-a"],
                    "api_b_max_inflight": max_inflight["api-b"],
                    "note": "per-process bulkhead ≠ cross-process global rate limit",
                },
                "lifespan_shutdown": {
                    "api_a_cleanup_logged": shutdown_a,
                    "api_b_cleanup_logged": shutdown_b,
                    "unclosed_client_warnings": unclosed,
                    "shutdown_tracebacks": shutdown_tracebacks,
                },
                "api_a_unexpected_restart_count": summary["container_identities"]["api-a"][
                    "unexpected_restart_count"
                ],
                "api_b_unexpected_restart_count": summary["container_identities"]["api-b"][
                    "unexpected_restart_count"
                ],
                "postgres_errors": postgres_errors,
                "redis_errors": redis_errors,
                "milvus_errors": milvus_errors,
                "post_identities": post_identities,
            }
        )

        errors: list[str] = []
        if summary["mode"] != "docker":
            errors.append("mode != docker")
        if summary["api_container_count"] != 2:
            errors.append("api_container_count != 2")
        if a_id == b_id:
            errors.append("container ids equal")
        if a_host == b_host:
            errors.append("hostnames equal")
        if summary["request_count"] != 20:
            errors.append("request_count != 20")
        if summary["api_a_count"] != 10 or summary["api_b_count"] != 10:
            errors.append("api split not 10/10")
        if summary.get("logical_provider_calls") != 20:
            errors.append(f"logical_provider_calls={summary.get('logical_provider_calls')}")
        if summary.get("physical_provider_attempts") != summary.get("stub_request_count"):
            errors.append(
                f"stub/physical mismatch stub={summary.get('stub_request_count')} "
                f"physical={summary.get('physical_provider_attempts')}"
            )
        ratio = float(summary.get("retry_amplification_ratio") or 0)
        if ratio <= 0 or ratio > 3.0:
            errors.append(f"amplification ratio out of range: {ratio}")
        if expected_failure != 1:
            errors.append(f"expected_failure_count={expected_failure}")
        if unexpected_failure != 0:
            errors.append(f"unexpected_failure={unexpected_failure}")
        if fallback != 2:
            errors.append(f"fallback_count={fallback} (expected 2)")
        if deadline_exceeded != 0:
            errors.append(f"deadline_exceeded={deadline_exceeded}")
        if context_leakage != 0:
            errors.append(f"context_leakage={context_leakage}")
        if duplicate_logical != 0:
            errors.append(f"duplicate_logical={duplicate_logical}")
        if cross_share:
            errors.append("client instance ids shared across containers")
        if len(client_ids["api-a"]) != 1 or len(client_ids["api-b"]) != 1:
            errors.append(
                f"client reuse failed a={sorted(client_ids['api-a'])} b={sorted(client_ids['api-b'])}"
            )
        if max_inflight["api-a"] > configured_limit or max_inflight["api-b"] > configured_limit:
            errors.append(f"inflight exceeded limit={configured_limit} seen={max_inflight}")
        if summary["api_a_unexpected_restart_count"] or summary["api_b_unexpected_restart_count"]:
            errors.append("unexpected API container restarts")
        if postgres_errors or redis_errors or milvus_errors:
            errors.append(
                f"dep errors pg={postgres_errors} redis={redis_errors} milvus={milvus_errors}"
            )
        if unclosed:
            errors.append(f"unclosed warnings={unclosed}")
        if shutdown_tracebacks:
            errors.append(f"shutdown_tracebacks={shutdown_tracebacks}")
        if not shutdown_a or not shutdown_b:
            errors.append("lifespan cleanup log missing on api-a/api-b")

        summary["errors"] = errors
        summary["status"] = "pass" if not errors else "fail"
        exit_code = 0 if not errors else 1

        (run_dir / "requests.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (docker_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (OUTPUT_ROOT / "docker_dual_api_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        print(
            json.dumps(
                {
                    k: summary[k]
                    for k in (
                        "status",
                        "mode",
                        "request_count",
                        "api_a_count",
                        "api_b_count",
                        "logical_provider_calls",
                        "physical_provider_attempts",
                        "stub_request_count",
                        "retry_amplification_ratio",
                        "success_count",
                        "degraded_count",
                        "expected_failure_count",
                        "unexpected_failure_count",
                        "fallback_count",
                        "provider_context_leakage_count",
                        "duplicate_logical_call_id_count",
                        "p50_ms",
                        "p95_ms",
                        "max_latency_ms",
                        "bulkhead",
                        "lifespan_shutdown",
                        "http_client_reuse",
                        "errors",
                    )
                    if k in summary
                },
                indent=2,
            )
        )
        print("api_a:", summary["api_a"])
        print("api_b:", summary["api_b"])

    except Exception as exc:  # noqa: BLE001
        summary["status"] = "fail"
        summary["blocked"] = str(exc)
        summary["errors"] = list(summary.get("errors") or []) + [str(exc)]
        exit_code = 1
        print(f"Docker Scenario G failed: {exc}", file=sys.stderr)
        try:
            _save_diagnostics(settings, docker_dir)
        except Exception:
            pass
        (docker_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    finally:
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        (docker_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if not args.keep and not args.skip_infra:
            print("11/11 cleaning compose stack...")
            _run_compose(settings, "down", "--remove-orphans", "-v", check=False)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
