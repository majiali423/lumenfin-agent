#!/usr/bin/env python3
"""Phase 3.3A deterministic provider resilience suite (local fault stub)."""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from provider_stub.server import serve, reset_state, snapshot

from lumenfin.llm import DeepSeekChatClient, LLMSettings, ResilientLLMClient, LocalFallbackLLMClient
from lumenfin.provider_resilience import (
    DeadlineExceededError,
    InvalidProviderResponseError,
    ProviderCallContext,
    summarize_provider_trace,
)
from lumenfin.rag.embeddings import DashScopeEmbeddingProvider, ResilientEmbeddingProvider

OUTPUT_DIR = ROOT / "outputs" / "phase33a_provider_resilience"
STUB_PORT = int(os.getenv("PHASE33A_STUB_PORT", "18090"))
STUB_BASE = f"http://127.0.0.1:{STUB_PORT}"


def _chat_client(*, scenario: str, max_attempts: int = 3, timeout: float = 2.0) -> DeepSeekChatClient:
    settings = LLMSettings(
        api_key="stub-key",
        base_url=STUB_BASE + "/v1",
        model="stub-model",
        timeout_seconds=timeout,
        max_retries=max_attempts,
        retry_backoff_seconds=0.05,
    )
    client = DeepSeekChatClient(settings)
    client.extra_headers["X-LumenFin-Scenario"] = scenario
    return client


def scenario_a(run_dir: Path) -> dict:
    reset_state()
    client = _chat_client(scenario="503_then_success", max_attempts=3)
    ctx = ProviderCallContext.create(request_id="a", deadline_seconds=30)
    ctx.sleep = lambda _: None
    client.bind_call_context(ctx)
    text = client.chat("sys", "user")
    out = {
        "status": "pass" if text.startswith("recovered") and client.last_attempts == 3 else "fail",
        "text": text,
        "attempts": client.last_attempts,
        "logical_calls": 1,
        "physical_attempts": client.last_attempts,
        "fallback": False,
        "stub": snapshot(),
    }
    if out["status"] != "pass":
        out["errors"] = ["503_then_success assertions failed"]
    (run_dir / "scenario_a.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def scenario_b(run_dir: Path) -> dict:
    reset_state()
    client = _chat_client(scenario="429_then_success", max_attempts=3)
    sleeps: list[float] = []
    ctx = ProviderCallContext.create(request_id="b", deadline_seconds=30)
    ctx.sleep = sleeps.append
    client.bind_call_context(ctx)
    text = client.chat("sys", "user")
    retry_events = [e for e in (ctx.trace_sink or []) if e.get("retry_after_ms")]
    out = {
        "status": "pass"
        if text.startswith("recovered") and retry_events and client.last_attempts == 2
        else "fail",
        "text": text,
        "attempts": client.last_attempts,
        "retry_after_seen": bool(retry_events),
        "sleeps": sleeps,
        "stub": snapshot(),
    }
    if out["status"] != "pass":
        out["errors"] = ["429_then_success assertions failed"]
    (run_dir / "scenario_b.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def scenario_c(run_dir: Path) -> dict:
    reset_state()
    client = _chat_client(scenario="permanent_400", max_attempts=3)
    ctx = ProviderCallContext.create(request_id="c", deadline_seconds=30)
    ctx.sleep = lambda _: None
    client.bind_call_context(ctx)
    error_class = None
    try:
        client.chat("sys", "user")
        status = "fail"
        errors = ["expected HTTP 400"]
    except Exception as exc:  # noqa: BLE001
        from lumenfin.provider_resilience import classify_provider_exception

        error_class = classify_provider_exception(exc)
        status = "pass" if client.last_attempts == 1 and error_class == "client_error" else "fail"
        errors = [] if status == "pass" else [f"unexpected {error_class} attempts={client.last_attempts}"]
    out = {
        "status": status,
        "attempts": client.last_attempts,
        "error_class": error_class,
        "errors": errors,
        "stub": snapshot(),
    }
    (run_dir / "scenario_c.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def scenario_d(run_dir: Path) -> dict:
    reset_state()
    client = _chat_client(scenario="slow_success", max_attempts=5, timeout=0.4)
    ctx = ProviderCallContext.create(request_id="d", deadline_seconds=0.8)
    ctx.sleep = lambda _: None
    client.bind_call_context(ctx)
    started = time.monotonic()
    error_class = None
    try:
        client.chat("sys", "user")
        status = "fail"
        errors = ["expected deadline exceeded"]
    except Exception as exc:  # noqa: BLE001
        from lumenfin.provider_resilience import classify_provider_exception

        error_class = classify_provider_exception(exc)
        elapsed = time.monotonic() - started
        status = "pass" if error_class in {"deadline_exceeded", "timeout"} and elapsed < 5.0 else "fail"
        errors = [] if status == "pass" else [f"{error_class} elapsed={elapsed}"]
    out = {
        "status": status,
        "error_class": error_class,
        "attempts": client.last_attempts,
        "errors": errors,
        "stub": snapshot(),
    }
    (run_dir / "scenario_d.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def scenario_e(run_dir: Path) -> dict:
    reset_state()
    primary = _chat_client(scenario="always_503", max_attempts=3)
    ctx = ProviderCallContext.create(request_id="e", deadline_seconds=30)
    ctx.sleep = lambda _: None
    primary.bind_call_context(ctx)
    resilient = ResilientLLMClient(primary=primary, fallback=LocalFallbackLLMClient(), allow_fallback=True)
    text = resilient.chat("sys", "Analyze NVIDIA briefly.")
    audit = resilient.fallback_audit()
    out = {
        "status": "pass"
        if resilient.used_fallback and resilient.degraded and audit.get("error_class")
        else "fail",
        "text_preview": text[:80],
        "audit": audit,
        "attempts": resilient.primary_attempts or primary.last_attempts,
        "stub": snapshot(),
    }
    if out["status"] != "pass":
        out["errors"] = ["fallback assertions failed"]
    (run_dir / "scenario_e.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def scenario_f(run_dir: Path) -> dict:
    reset_state()
    inner = DashScopeEmbeddingProvider(
        api_key="stub",
        dimension=64,
        base_url=STUB_BASE + "/v1",
        timeout_seconds=2.0,
        client=None,
    )
    # Inject scenario via dedicated client to avoid shared-post races.
    import httpx

    http = httpx.Client(timeout=2.0, trust_env=False)
    real_post = http.post

    def post(url, headers=None, json=None, timeout=None):
        headers = dict(headers or {})
        headers["X-LumenFin-Scenario"] = "embedding_dimension_mismatch"
        return real_post(url, headers=headers, json=json, timeout=timeout)

    http.post = post  # type: ignore[method-assign]
    inner._client = http
    provider = ResilientEmbeddingProvider(inner, max_retries=3, backoff_seconds=0.01, sleep=lambda _: None)
    error_class = None
    try:
        provider.embed(["hello", "world"])
        status = "fail"
        errors = ["expected invalid_response"]
    except Exception as exc:  # noqa: BLE001
        from lumenfin.provider_resilience import classify_provider_exception

        error_class = classify_provider_exception(exc)
        status = "pass" if error_class == "invalid_response" and provider.last_attempts == 1 else "fail"
        errors = [] if status == "pass" else [f"{error_class} attempts={provider.last_attempts}"]
    finally:
        http.close()
    out = {
        "status": status,
        "error_class": error_class,
        "attempts": provider.last_attempts,
        "errors": errors,
        "stub": snapshot(),
    }
    (run_dir / "scenario_f.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def scenario_g(run_dir: Path) -> dict:
    reset_state()
    scenarios = (
        ["success"] * 6
        + ["503_then_success"] * 2
        + ["429_then_success"]
        + ["always_503"]
    )
    latencies: list[float] = []
    success = degraded = failure = 0
    traces: list[dict] = []

    def one(scenario: str) -> dict:
        started = time.perf_counter()
        if scenario == "always_503":
            primary = _chat_client(scenario=scenario, max_attempts=2)
            ctx = ProviderCallContext.create(deadline_seconds=20)
            ctx.sleep = lambda _: None
            primary.bind_call_context(ctx)
            client = ResilientLLMClient(primary=primary, fallback=LocalFallbackLLMClient(), allow_fallback=True)
            text = client.chat("sys", "user")
            elapsed = (time.perf_counter() - started) * 1000
            return {
                "ok": True,
                "degraded": client.degraded,
                "latency_ms": elapsed,
                "trace": list(getattr(primary, "last_trace", []) or []),
                "fallback": client.used_fallback,
            }
        client = _chat_client(scenario=scenario, max_attempts=3)
        ctx = ProviderCallContext.create(deadline_seconds=20)
        ctx.sleep = lambda _: None
        client.bind_call_context(ctx)
        text = client.chat("sys", "user")
        elapsed = (time.perf_counter() - started) * 1000
        return {
            "ok": bool(text),
            "degraded": False,
            "latency_ms": elapsed,
            "trace": list(client.last_trace or []),
            "fallback": False,
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(one, s) for s in scenarios]
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception:
                failure += 1
                continue
            latencies.append(row["latency_ms"])
            traces.extend(row["trace"])
            if row["degraded"] or row["fallback"]:
                degraded += 1
            elif row["ok"]:
                success += 1
            else:
                failure += 1

    summary = summarize_provider_trace(traces)
    stub = snapshot()
    latencies_sorted = sorted(latencies)
    p50 = statistics.median(latencies_sorted) if latencies_sorted else 0
    p95 = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))] if latencies_sorted else 0
    out = {
        "status": "pass"
        if failure == 0 and stub["request_count"] > 0 and summary["retry_amplification_ratio"] <= 3.0
        else "fail",
        "request_count": len(scenarios),
        "success_count": success,
        "degraded_count": degraded,
        "failure_count": failure,
        "logical_provider_calls": summary["logical_provider_calls"],
        "physical_provider_attempts": summary["physical_provider_attempts"],
        "retry_amplification_ratio": summary["retry_amplification_ratio"],
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "max_latency_ms": round(max(latencies_sorted), 2) if latencies_sorted else 0,
        "fallback_count": degraded,
        "deadline_exceeded_count": summary["deadline_exceeded"],
        "unexpected_error_count": failure,
        "stub_request_count": stub["request_count"],
    }
    if out["status"] != "pass":
        out["errors"] = ["concurrency scenario assertions failed"]
    (run_dir / "scenario_g.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = OUTPUT_DIR / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    server = serve("127.0.0.1", STUB_PORT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    summary: dict = {
        "phase": "3.3A",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stub_base": STUB_BASE,
        "live_smoke": "skipped",
        "scenario_status": {},
        "unexpected_error_count": 0,
    }
    exit_code = 0
    try:
        for name, fn in [
            ("A_503_then_success", scenario_a),
            ("B_429_retry_after", scenario_b),
            ("C_permanent_400", scenario_c),
            ("D_deadline", scenario_d),
            ("E_fallback", scenario_e),
            ("F_embedding_invalid", scenario_f),
            ("G_concurrency", scenario_g),
        ]:
            print(f"Running {name}...")
            result = fn(run_dir)
            summary["scenario_status"][name] = result
            if result.get("status") != "pass":
                exit_code = 1
                summary["unexpected_error_count"] += len(result.get("errors") or []) or 1

        g = summary["scenario_status"].get("G_concurrency") or {}
        summary.update(
            {
                "logical_provider_calls": g.get("logical_provider_calls"),
                "physical_provider_attempts": g.get("physical_provider_attempts"),
                "retry_amplification_ratio": g.get("retry_amplification_ratio"),
                "p50_ms": g.get("p50_ms"),
                "p95_ms": g.get("p95_ms"),
                "max_latency_ms": g.get("max_latency_ms"),
                "fallback_count": g.get("fallback_count"),
                "deadline_exceeded_count": g.get("deadline_exceeded_count"),
            }
        )
        failed = [
            name
            for name, payload in summary["scenario_status"].items()
            if payload.get("status") != "pass"
        ]
        summary["status"] = "pass" if not failed else "fail"
        summary["failed_scenarios"] = failed
        # Persist stub stats as provider_calls.jsonl
        with (run_dir / "provider_calls.jsonl").open("w", encoding="utf-8") as fh:
            for item in snapshot().get("requests") or []:
                fh.write(json.dumps(item) + "\n")
        (run_dir / "provider-stub.log").write_text(
            json.dumps(snapshot(), indent=2), encoding="utf-8"
        )
    except Exception as exc:  # noqa: BLE001
        summary["status"] = "fail"
        summary["blocked"] = str(exc)
        exit_code = 1
    finally:
        server.shutdown()
        server.server_close()
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({k: summary[k] for k in (
            "status",
            "failed_scenarios",
            "logical_provider_calls",
            "physical_provider_attempts",
            "retry_amplification_ratio",
            "p50_ms",
            "p95_ms",
            "max_latency_ms",
            "fallback_count",
            "deadline_exceeded_count",
            "live_smoke",
            "unexpected_error_count",
        ) if k in summary}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
