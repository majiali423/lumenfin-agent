"""Prometheus operational metrics; never export user or evidence identifiers."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Any, TypeVar

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from prometheus_client.exposition import start_http_server
from redis import Redis
from redis.exceptions import RedisError
from starlette.requests import Request
from starlette.responses import Response

F = TypeVar("F", bound=Callable[..., Any])

HTTP_REQUESTS = Counter(
    "lumenfin_http_requests_total", "HTTP requests handled by the API.",
    ("method", "route", "status_class"),
)
HTTP_DURATION = Histogram(
    "lumenfin_http_request_duration_seconds", "API request latency.",
    ("method", "route"),
    buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 180),
)
HTTP_INFLIGHT = Gauge(
    "lumenfin_http_inflight_requests", "HTTP requests currently executing."
)
WORKFLOW_RUNS = Counter(
    "lumenfin_workflow_runs_total", "Workflow terminal outcomes.",
    ("operation", "status"),
)
WORKFLOW_DURATION = Histogram(
    "lumenfin_workflow_duration_seconds", "End-to-end service workflow latency.",
    ("operation", "status"),
    buckets=(.05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120, 180, 300),
)
WORKFLOW_ERRORS = Counter(
    "lumenfin_workflow_errors_total", "Workflow exceptions by class.",
    ("operation", "error_type"),
)
NODE_DURATION = Histogram(
    "lumenfin_node_duration_seconds", "Recorded LangGraph node latency.",
    ("step", "status"),
    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60),
)
REPAIR_ROUTES = Counter(
    "lumenfin_repair_routes_total", "Bounded repair targets selected by the critic.",
    ("target",),
)
PROVIDER_CALLS = Counter(
    "lumenfin_provider_calls_total", "Logical provider calls.",
    ("provider", "operation", "outcome"),
)
PROVIDER_RETRIES = Counter(
    "lumenfin_provider_retries_total", "Provider retries.",
    ("provider", "operation"),
)
PROVIDER_FALLBACKS = Counter(
    "lumenfin_provider_fallbacks_total", "Provider fallbacks.",
    ("provider", "operation"),
)
PROVIDER_BATCH_DURATION = Histogram(
    "lumenfin_provider_batch_duration_seconds",
    "Total provider latency recorded for one workflow.",
    ("provider", "operation", "outcome"),
    buckets=(.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60, 120),
)
RAG_QUERIES = Counter(
    "lumenfin_rag_queries_total", "RAG executions by path and outcome.",
    ("path", "outcome"),
)
RAG_RETRIEVAL_DURATION = Histogram(
    "lumenfin_rag_retrieval_duration_seconds", "Recorded retrieval-node latency.",
    ("path", "outcome"),
    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30, 60),
)
RERANK_DURATION = Histogram(
    "lumenfin_rerank_duration_seconds", "Reranker latency from RAG telemetry.",
    ("provider", "outcome"),
    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 30),
)
RERANK_FALLBACKS = Counter(
    "lumenfin_rerank_fallbacks_total", "Reranker fallbacks.", ("provider",)
)
CITATION_RESULTS = Counter(
    "lumenfin_citation_validation_total",
    "Structured citation contract outcomes; not citation accuracy.", ("status",),
)
QUEUE_DEPTH = Gauge(
    "lumenfin_queue_depth", "Redis queue depth observed during an API scrape.",
    ("queue", "state"),
)
QUEUE_OBSERVATION_ERRORS = Counter(
    "lumenfin_queue_observation_errors_total", "Queue depth collection failures.",
    ("queue",),
)
QUEUE_EVENTS = Counter(
    "lumenfin_queue_events_total", "Reliable queue lifecycle events.",
    ("queue", "event"),
)
WORKER_JOBS = Counter(
    "lumenfin_worker_jobs_total", "Worker job outcomes.", ("queue", "action")
)
PROCESS_START_TIME = Gauge(
    "lumenfin_process_start_time_seconds", "Instrumented process start time.", ("role",)
)

_SERVER_LOCK = threading.Lock()
_STARTED_PORTS: set[int] = set()


def prometheus_enabled() -> bool:
    return os.getenv("MAS_PROMETHEUS_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off"
    }


def instrument_workflow(operation: str) -> Callable[[F], F]:
    """Measure one synchronous service workflow without changing its result."""

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            if not prometheus_enabled():
                return func(*args, **kwargs)
            started = time.perf_counter()
            try:
                payload = func(*args, **kwargs)
            except Exception as exc:
                elapsed = max(0.0, time.perf_counter() - started)
                WORKFLOW_RUNS.labels(operation=operation, status="error").inc()
                WORKFLOW_DURATION.labels(operation=operation, status="error").observe(elapsed)
                WORKFLOW_ERRORS.labels(operation=operation, error_type=type(exc).__name__).inc()
                raise
            observe_packaged_run(
                payload, operation=operation,
                duration_seconds=max(0.0, time.perf_counter() - started),
            )
            return payload

        return wrapped  # type: ignore[return-value]
    return decorator


def observe_packaged_run(
    payload: dict[str, Any], *, operation: str, duration_seconds: float
) -> None:
    """Project existing run_telemetry into bounded Prometheus aggregates."""

    status = _label(payload.get("workflow_status"), "unknown")
    WORKFLOW_RUNS.labels(operation=operation, status=status).inc()
    WORKFLOW_DURATION.labels(operation=operation, status=status).observe(duration_seconds)
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    telemetry = result.get("run_telemetry") if isinstance(result.get("run_telemetry"), dict) else {}
    spans = telemetry.get("node_spans") if isinstance(telemetry.get("node_spans"), list) else []
    retrieval_seconds = 0.0
    for span in spans:
        if not isinstance(span, dict):
            continue
        step = _label(span.get("step"), "unknown")
        span_status = _label(span.get("status"), "unknown")
        seconds = max(0.0, _number(span.get("latency_ms")) / 1000.0)
        NODE_DURATION.labels(step=step, status=span_status).observe(seconds)
        if step == "retrieval":
            retrieval_seconds += seconds
    target = _label(result.get("critic_repair_target"), "")
    if target:
        REPAIR_ROUTES.labels(target=target).inc()

    degraded = payload.get("provider_degraded")
    provider = _label(
        degraded.get("provider") if isinstance(degraded, dict) else payload.get("llm_backend"),
        "unknown",
    )
    summary = payload.get("provider_call_summary") if isinstance(payload.get("provider_call_summary"), dict) else {}
    calls = max(0, int(_number(summary.get("logical_provider_calls"))))
    successes = max(0, min(calls, int(_number(summary.get("successes")))))
    failures = max(0, calls - successes)
    if successes:
        PROVIDER_CALLS.labels(provider=provider, operation="chat", outcome="success").inc(successes)
    if failures:
        PROVIDER_CALLS.labels(provider=provider, operation="chat", outcome="error").inc(failures)
    retries = max(0, int(_number(summary.get("retries"))))
    fallbacks = max(0, int(_number(summary.get("fallbacks"))))
    if retries:
        PROVIDER_RETRIES.labels(provider=provider, operation="chat").inc(retries)
    if fallbacks:
        PROVIDER_FALLBACKS.labels(provider=provider, operation="chat").inc(fallbacks)
    if calls:
        outcome = "error" if failures else "success"
        PROVIDER_BATCH_DURATION.labels(
            provider=provider, operation="chat", outcome=outcome
        ).observe(max(0.0, _number(summary.get("total_provider_latency_ms")) / 1000.0))

    rag = telemetry.get("rag") if isinstance(telemetry.get("rag"), dict) else {}
    if rag:
        path = _label(rag.get("mode"), "unknown")
        outcome = "degraded" if bool(rag.get("degraded")) else "ok"
        RAG_QUERIES.labels(path=path, outcome=outcome).inc()
        RAG_RETRIEVAL_DURATION.labels(path=path, outcome=outcome).observe(retrieval_seconds)
        providers = rag.get("rerank_providers") if isinstance(rag.get("rerank_providers"), list) else []
        reranker = _label(providers[0] if providers else "none", "none")
        fallback_count = max(0, int(_number(rag.get("rerank_fallbacks"))))
        rerank_outcome = "fallback" if fallback_count else outcome
        RERANK_DURATION.labels(provider=reranker, outcome=rerank_outcome).observe(
            max(0.0, _number(rag.get("rerank_latency_ms")) / 1000.0)
        )
        if fallback_count:
            RERANK_FALLBACKS.labels(provider=reranker).inc(fallback_count)
        company_count = max(0, int(_number(rag.get("company_count"))))
        if reranker != "none" and company_count:
            successful_reranks = max(0, company_count - fallback_count)
            if successful_reranks:
                PROVIDER_CALLS.labels(
                    provider=reranker, operation="rerank", outcome="success"
                ).inc(successful_reranks)
            if fallback_count:
                PROVIDER_CALLS.labels(
                    provider=reranker, operation="rerank", outcome="error"
                ).inc(fallback_count)
                PROVIDER_FALLBACKS.labels(provider=reranker, operation="rerank").inc(
                    fallback_count
                )
            attempts = max(0, int(_number(rag.get("rerank_attempts"))))
            extra_attempts = max(0, attempts - company_count)
            if extra_attempts:
                PROVIDER_RETRIES.labels(provider=reranker, operation="rerank").inc(
                    extra_attempts
                )
    CITATION_RESULTS.labels(status=_citation_status(result)).inc()


async def prometheus_http_middleware(
    request: Request, call_next: Callable[..., Any]
) -> Response:
    if not prometheus_enabled() or request.url.path == "/metrics":
        return await call_next(request)
    HTTP_INFLIGHT.inc()
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = int(response.status_code)
        return response
    finally:
        route = request.scope.get("route")
        route_path = getattr(route, "path", None) or _fallback_route(request.url.path)
        method = _label(request.method.upper(), "UNKNOWN")
        HTTP_REQUESTS.labels(
            method=method, route=route_path, status_class=f"{status_code // 100}xx"
        ).inc()
        HTTP_DURATION.labels(method=method, route=route_path).observe(
            max(0.0, time.perf_counter() - started)
        )
        HTTP_INFLIGHT.dec()


def render_metrics(*, redis_url: str | None = None, queue_names: Iterable[str] = ()) -> Response:
    if redis_url:
        refresh_queue_depths(redis_url, queue_names)
    return Response(content=generate_latest(), headers={"Content-Type": CONTENT_TYPE_LATEST})


def refresh_queue_depths(redis_url: str, queue_names: Iterable[str]) -> None:
    for queue_name in sorted({_label(name, "") for name in queue_names if name}):
        client: Redis | None = None
        try:
            client = Redis.from_url(redis_url, socket_connect_timeout=.5, socket_timeout=.5)
            pipe = client.pipeline(transaction=False)
            states = ("pending", "processing", "dead-letter", "legacy")
            pipe.llen(f"{queue_name}:pending")
            pipe.llen(f"{queue_name}:processing")
            pipe.llen(f"{queue_name}:dead-letter")
            pipe.llen(queue_name)
            for state, value in zip(states, pipe.execute(), strict=True):
                QUEUE_DEPTH.labels(queue=queue_name, state=state).set(max(0, int(value or 0)))
        except (RedisError, TimeoutError, ConnectionError, OSError, ValueError):
            QUEUE_OBSERVATION_ERRORS.labels(queue=queue_name).inc()
        finally:
            if client is not None:
                try:
                    client.close()
                except (RedisError, OSError):
                    pass


def record_queue_event(queue: str, event: str, count: int = 1) -> None:
    if prometheus_enabled() and count > 0:
        QUEUE_EVENTS.labels(queue=_label(queue, "unknown"), event=_label(event, "unknown")).inc(count)


def record_worker_job(queue: str, action: str) -> None:
    if prometheus_enabled():
        WORKER_JOBS.labels(queue=_label(queue, "unknown"), action=_label(action, "unknown")).inc()


def start_worker_metrics_server(role: str, *, port: int | None = None) -> bool:
    if not prometheus_enabled():
        return False
    resolved = port if port is not None else int(os.getenv("MAS_METRICS_PORT", "0") or 0)
    if resolved <= 0:
        return False
    with _SERVER_LOCK:
        if resolved in _STARTED_PORTS:
            return True
        start_http_server(resolved, addr="0.0.0.0")
        _STARTED_PORTS.add(resolved)
        PROCESS_START_TIME.labels(role=_label(role, "worker")).set(time.time())
    return True


def _citation_status(result: dict[str, Any]) -> str:
    structured = result.get("structured_answer")
    if not isinstance(structured, dict):
        return "unavailable"
    validation = str(structured.get("citation_validation") or "").strip().lower()
    if validation == "failed":
        return "validation_failed"
    citations = structured.get("citations")
    if validation == "passed" and isinstance(citations, list) and citations:
        return "valid"
    return "unavailable"


def _fallback_route(path: str) -> str:
    if path.startswith("/static/"):
        return "/static/{path}"
    return path if path in {"/", "/health", "/ready"} else "unmatched"


def _label(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text[:80] if text else default


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
