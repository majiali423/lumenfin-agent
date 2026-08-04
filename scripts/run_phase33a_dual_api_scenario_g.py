#!/usr/bin/env python3
"""Phase 3.3A dual-API Scenario G: two OS processes + shared provider stub."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from provider_stub.server import serve, reset_state, snapshot

OUTPUT_DIR = ROOT / "outputs" / "phase33a_provider_resilience"
STUB_PORT = int(os.getenv("PHASE33A_STUB_PORT", "18090"))
API_A_PORT = int(os.getenv("PHASE33A_API_A_PORT", "18180"))
API_B_PORT = int(os.getenv("PHASE33A_API_B_PORT", "18181"))


def _wait_health(url: str, *, timeout: float = 60.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=2.0, trust_env=False)
            if resp.status_code == 200:
                return resp.json()
            last = resp.text
        except Exception as exc:  # noqa: BLE001
            last = str(exc)
        time.sleep(0.4)
    raise RuntimeError(f"health timeout for {url}: {last}")


def _spawn_api(port: int, worker_id: str, stub_base: str) -> subprocess.Popen:
    env = os.environ.copy()
    # Avoid EnvConflictError: do not override credential keys present in .env.
    for key in (
        "DEEPSEEK_API_KEY",
        "DASHSCOPE_API_KEY",
        "MAS_API_KEY",
        "OPENAI_API_KEY",
    ):
        env.pop(key, None)
    # Ensure a non-empty DeepSeek key for provider registry without conflicting .env:
    # if .env lacks a key, inject stub-key only when absent after pop.
    dotenv_path = ROOT / ".env"
    has_dotenv_key = False
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("DEEPSEEK_API_KEY=") and len(line.strip()) > len("DEEPSEEK_API_KEY="):
                has_dotenv_key = True
                break
    env.update(
        {
            "APP_ENV": "test",
            "MAS_ALLOW_SQLITE_DEV": "true",
            "MAS_WORKER_ID": worker_id,
            "MAS_HOST": "127.0.0.1",
            "MAS_PORT": str(port),
            "DEEPSEEK_BASE_URL": f"{stub_base}/v1",
            "DEEPSEEK_MAX_RETRIES": "3",
            "DEEPSEEK_TIMEOUT_SECONDS": "3",
            "DEEPSEEK_RETRY_BACKOFF_SECONDS": "0.05",
            "ALLOW_LOCAL_FALLBACK": "true",
            "MAS_ANALYSIS_DEADLINE_SECONDS": "20",
            "MAS_FETCH_LIVE_FUNDAMENTALS": "false",
            "MAS_FETCH_SEC_FUNDAMENTALS": "false",
            "MAS_RAG_ENABLED": "false",
            "MAS_EMBEDDING_PROVIDER": "deterministic",
            "PYTHONPATH": str(SRC),
            "PYTHONUNBUFFERED": "1",
        }
    )
    if not has_dotenv_key:
        env["DEEPSEEK_API_KEY"] = "stub-key"
    # Isolated sqlite paths per API process.
    data_dir = ROOT / "outputs" / "phase33a_provider_resilience" / "_dual_api_runtime" / worker_id
    data_dir.mkdir(parents=True, exist_ok=True)
    env["MAS_DB_PATH"] = str(data_dir / "app.db")
    env["MAS_OUTPUT_DIR"] = str(data_dir / "outputs")
    env["MAS_UPLOAD_DIR"] = str(data_dir / "uploads")
    log_path = data_dir / "uvicorn.log"
    cmd = [
        sys.executable,
        "-c",
        (
            "from lumenfin.api.app import create_app; "
            "import uvicorn; "
            f"uvicorn.run(create_app(), host='127.0.0.1', port={port}, log_level='warning')"
        ),
    ]
    log_fh = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
    )


def _fault_plan() -> list[tuple[str, str]]:
    """Return (api, scenario) pairs: 10→A + 10→B with required fault mix."""
    # 50% success, 20% 503_then_success, 15% 429_then_success, 10% timeout/fallback, 5% permanent 400
    scenarios = (
        ["success"] * 10
        + ["503_then_success"] * 4
        + ["429_then_success"] * 3
        + ["always_503"] * 2
        + ["permanent_400"] * 1
    )
    assert len(scenarios) == 20
    plan = []
    for idx, scenario in enumerate(scenarios):
        api = "api-a" if idx < 10 else "api-b"
        plan.append((api, scenario))
    return plan


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3.3A dual-API Scenario G")
    parser.add_argument(
        "--mode",
        choices=("process", "docker"),
        default="process",
        help="process=local OS subprocesses (default); docker=real containers via run_phase33a_dual_api_docker.py",
    )
    args, unknown = parser.parse_known_args(argv)
    if args.mode == "docker":
        # Explicit Docker mode only — never silently fall back to OS processes.
        import run_phase33a_dual_api_docker as docker_runner

        # Forward remaining args to the Docker runner (e.g. --keep).
        return docker_runner.main()
    if unknown:
        print(f"Ignoring unknown args in process mode: {unknown}", file=sys.stderr)
    return _run_process_mode()


def _run_process_mode() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_DIR / f"dual_api_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    stub_base = f"http://127.0.0.1:{STUB_PORT}"
    reset_state()
    server = serve("127.0.0.1", STUB_PORT)
    stub_thread = threading.Thread(target=server.serve_forever, daemon=True)
    stub_thread.start()
    time.sleep(0.2)

    proc_a = _spawn_api(API_A_PORT, "api-a", stub_base)
    proc_b = _spawn_api(API_B_PORT, "api-b", stub_base)
    exit_code = 0
    summary: dict = {
        "phase": "3.3A",
        "scenario": "G_dual_api",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "mode": "dual_os_process",
        "note": "per-process bulkhead ≠ cross-process global rate limit",
    }
    try:
        health_a = _wait_health(f"http://127.0.0.1:{API_A_PORT}/health")
        health_b = _wait_health(f"http://127.0.0.1:{API_B_PORT}/health")
        summary["api_a"] = {
            "url": f"http://127.0.0.1:{API_A_PORT}",
            "pid": health_a.get("pid") or proc_a.pid,
            "worker_id": health_a.get("worker_id") or "api-a",
            "container_id": None,
            "process_id": proc_a.pid,
        }
        summary["api_b"] = {
            "url": f"http://127.0.0.1:{API_B_PORT}",
            "pid": health_b.get("pid") or proc_b.pid,
            "worker_id": health_b.get("worker_id") or "api-b",
            "container_id": None,
            "process_id": proc_b.pid,
        }
        if summary["api_a"]["pid"] == summary["api_b"]["pid"]:
            raise RuntimeError("API A and API B must be different OS processes")

        plan = _fault_plan()
        bases = {
            "api-a": f"http://127.0.0.1:{API_A_PORT}",
            "api-b": f"http://127.0.0.1:{API_B_PORT}",
        }
        latencies: list[float] = []
        rows: list[dict] = []
        traces: list[dict] = []
        client_ids: dict[str, set[str]] = {"api-a": set(), "api-b": set()}
        request_ids: list[str] = []
        logical_ids: list[str] = []

        def one(api: str, scenario: str) -> dict:
            started = time.perf_counter()
            with httpx.Client(timeout=30.0, trust_env=False) as client:
                resp = client.post(
                    f"{bases[api]}/api/v1/provider-resilience/probe",
                    json={
                        "scenario": scenario,
                        "prompt": f"{api}:{scenario}:{uuid4().hex[:6]}",
                        "max_attempts": 3 if scenario != "always_503" else 2,
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
        for row in rows:
            api_counts[row["api"]] += 1
            latencies.append(row["latency_ms"])
            body = row["body"] or {}
            request_ids.append(str(body.get("request_id") or ""))
            if body.get("client_id"):
                client_ids[row["api"]].add(str(body["client_id"]))
            for event in body.get("trace") or []:
                traces.append(event)
                lid = event.get("logical_call_id")
                if lid:
                    logical_ids.append(str(lid))
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
        stub = snapshot()
        latencies_sorted = sorted(latencies)
        p50 = statistics.median(latencies_sorted) if latencies_sorted else 0
        p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))] if latencies_sorted else 0

        # Shared client objects must not cross process boundaries.
        cross_share = client_ids["api-a"] & client_ids["api-b"]
        unique_logical = len(set(logical_ids))
        unique_requests = len({r for r in request_ids if r})
        logical_to_request: dict[str, str] = {}
        logical_cross_request = False
        for event in traces:
            lid = str(event.get("logical_call_id") or "")
            rid = str(event.get("request_id") or "")
            if not lid or not rid:
                continue
            prior = logical_to_request.get(lid)
            if prior is not None and prior != rid:
                logical_cross_request = True
                break
            logical_to_request[lid] = rid

        summary.update(
            {
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
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "max_latency_ms": round(max(latencies_sorted), 2) if latencies_sorted else 0,
                "stub_request_count": stub.get("request_count"),
                "unique_request_ids": unique_requests,
                "unique_logical_call_ids": unique_logical,
                "client_ids_by_api": {k: sorted(v) for k, v in client_ids.items()},
                "cross_process_shared_client_ids": sorted(cross_share),
                "provider_stub": stub,
            }
        )

        errors = []
        if summary["request_count"] != 20:
            errors.append("request_count != 20")
        if summary["api_a_count"] != 10 or summary["api_b_count"] != 10:
            errors.append("api split not 10/10")
        if unexpected_failure:
            errors.append(f"unexpected_failure={unexpected_failure}")
        if cross_share:
            errors.append("shared httpx client ids leaked across processes")
        if logical_cross_request:
            errors.append("logical_call_id reused across requests")
        if stub.get("request_count", 0) <= 0:
            errors.append("stub recorded no requests")
        if (trace_summary.get("retry_amplification_ratio") or 0) > 3.0:
            errors.append("amplification ratio > 3")
        if not client_ids["api-a"] or not client_ids["api-b"]:
            errors.append("both APIs must produce provider client calls")
        # Physical attempts should reconcile with stub request count for chat calls.
        if stub.get("request_count") != summary.get("physical_provider_attempts"):
            errors.append(
                f"stub/physical mismatch stub={stub.get('request_count')} "
                f"physical={summary.get('physical_provider_attempts')}"
            )

        summary["errors"] = errors
        summary["status"] = "pass" if not errors else "fail"
        if errors:
            exit_code = 1

        (run_dir / "requests.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "dual_api_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in (
            "status",
            "request_count",
            "api_a_count",
            "api_b_count",
            "logical_provider_calls",
            "physical_provider_attempts",
            "retry_amplification_ratio",
            "success_count",
            "degraded_count",
            "expected_failure_count",
            "unexpected_failure_count",
            "fallback_count",
            "deadline_exceeded_count",
            "provider_busy_count",
            "p50_ms",
            "p95_ms",
            "max_latency_ms",
            "stub_request_count",
            "errors",
        ) if k in summary}, indent=2))
        print("api_a:", summary["api_a"])
        print("api_b:", summary["api_b"])
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "fail"
        summary["blocked"] = str(exc)
        exit_code = 1
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    finally:
        for proc in (proc_a, proc_b):
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        server.shutdown()
        server.server_close()
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
