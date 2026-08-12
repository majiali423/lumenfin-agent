"""Unified provider call policy, request deadlines, and desensitized traces.

Semantics:
  max_attempts = total physical attempts including the first call
  (max_attempts=3 → up to 3 HTTP calls and at most 2 sleeps)

Request state lives only on ProviderCallContext (never module globals).
"""

from __future__ import annotations

import random
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, TypeVar

import httpx

from .provider_retry import (
    classify_exception,
    is_transient_error_class,
    is_transient_exception,
)

T = TypeVar("T")

_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|token|password|secret|access[_-]?key)"
    r"\s*[:=]\s*\S+"
)
_CREDENTIAL_URL_RE = re.compile(
    r"(?i)\b([a-z][a-z0-9+.-]*://[^:/@\s]+:)[^@\s]+(@)"
)


class DeadlineExceededError(TimeoutError):
    """Raised when the request-level deadline is exhausted."""

    error_class = "deadline_exceeded"


class ProviderBusyError(RuntimeError):
    """Raised when a per-process provider bulkhead cannot be acquired in time."""

    error_class = "provider_busy"


class InvalidProviderResponseError(ValueError):
    """Non-transient malformed or schema-invalid provider payload."""

    error_class = "invalid_response"


@dataclass(frozen=True)
class ProviderCallPolicy:
    provider: str
    operation: str
    max_attempts: int = 3
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 45.0
    write_timeout_seconds: float = 45.0
    pool_timeout_seconds: float = 5.0
    base_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 30.0
    jitter_ratio: float = 0.2

    def httpx_timeout(self, *, remaining_seconds: float | None = None) -> httpx.Timeout:
        read = self.read_timeout_seconds
        write = self.write_timeout_seconds
        connect = self.connect_timeout_seconds
        pool = self.pool_timeout_seconds
        if remaining_seconds is not None:
            budget = max(0.05, float(remaining_seconds))
            read = min(read, budget)
            write = min(write, budget)
            connect = min(connect, budget)
            pool = min(pool, budget)
        return httpx.Timeout(connect=connect, read=read, write=write, pool=pool)


@dataclass
class ProviderCallContext:
    request_id: str
    thread_id: str | None = None
    deadline_monotonic: float | None = None
    trace_sink: list[dict[str, Any]] | None = None
    rng: random.Random | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)
    now: Callable[[], float] = field(default=time.monotonic)

    @classmethod
    def create(
        cls,
        *,
        request_id: str | None = None,
        thread_id: str | None = None,
        deadline_seconds: float | None = None,
        trace_sink: list[dict[str, Any]] | None = None,
        rng: random.Random | None = None,
    ) -> "ProviderCallContext":
        now = time.monotonic()
        deadline = None if deadline_seconds is None else now + max(0.0, float(deadline_seconds))
        return cls(
            request_id=(request_id or uuid.uuid4().hex),
            thread_id=thread_id,
            deadline_monotonic=deadline,
            trace_sink=trace_sink if trace_sink is not None else [],
            rng=rng or random.Random(),
        )

    def remaining_seconds(self) -> float | None:
        if self.deadline_monotonic is None:
            return None
        return self.deadline_monotonic - self.now()

    def ensure_budget(self, *, minimum_seconds: float = 0.05) -> float | None:
        remaining = self.remaining_seconds()
        if remaining is None:
            return None
        if remaining < minimum_seconds:
            raise DeadlineExceededError(
                f"request deadline exceeded (remaining={remaining:.3f}s)"
            )
        return remaining


def redact_provider_message(message: str, *, limit: int = 300) -> str:
    text = _SECRET_RE.sub(r"\1=[REDACTED]", str(message or ""))
    text = _CREDENTIAL_URL_RE.sub(r"\1[REDACTED]\2", text)
    text = text.replace("\n", " ").strip()
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def extract_retry_after_seconds(exc: BaseException) -> float | None:
    """Parse Retry-After integer-seconds from an httpx HTTPStatusError, if present."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    raw = exc.response.headers.get("Retry-After") or exc.response.headers.get("retry-after")
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return max(0.0, float(int(text)))
    except ValueError:
        return None


def classify_provider_exception(exc: BaseException) -> str:
    if isinstance(exc, DeadlineExceededError):
        return "deadline_exceeded"
    if isinstance(exc, ProviderBusyError):
        return "provider_busy"
    if isinstance(exc, InvalidProviderResponseError):
        return "invalid_response"
    return classify_exception(exc)


def is_retryable_provider_exception(exc: BaseException) -> bool:
    if isinstance(exc, (DeadlineExceededError, ProviderBusyError, InvalidProviderResponseError)):
        return False
    return is_transient_exception(exc)


def compute_backoff_seconds(
    *,
    attempt_index: int,
    policy: ProviderCallPolicy,
    retry_after_seconds: float | None = None,
    remaining_seconds: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Bounded exponential backoff with optional Retry-After and jitter."""
    base = policy.base_backoff_seconds * (2 ** max(0, attempt_index))
    delay = min(policy.max_backoff_seconds, base)
    if retry_after_seconds is not None:
        delay = max(delay, float(retry_after_seconds))
        delay = min(delay, policy.max_backoff_seconds)
    jitter_ratio = max(0.0, min(1.0, policy.jitter_ratio))
    if jitter_ratio > 0:
        generator = rng or random.Random()
        # Bounded multiplicative jitter in [1-j, 1+j].
        factor = 1.0 + generator.uniform(-jitter_ratio, jitter_ratio)
        delay = max(0.0, delay * factor)
    if remaining_seconds is not None:
        # Leave a tiny slice so the next attempt can still start.
        delay = min(delay, max(0.0, remaining_seconds - 0.05))
    return delay


def summarize_provider_trace(events: list[dict[str, Any]]) -> dict[str, Any]:
    logical_ids = {str(item.get("logical_call_id")) for item in events if item.get("logical_call_id")}
    physical = sum(1 for item in events if item.get("status") in {"success", "retry", "error", "fallback"})
    # Count attempt rows (each physical try emits one row).
    attempts = [item for item in events if "attempt" in item]
    physical_attempts = len(attempts) if attempts else physical
    logical_calls = len(logical_ids) if logical_ids else (
        1 if events else 0
    )
    successes = sum(1 for item in events if item.get("status") == "success")
    retries = sum(1 for item in events if item.get("status") == "retry")
    timeouts = sum(1 for item in events if item.get("error_class") == "timeout")
    rate_limits = sum(1 for item in events if item.get("error_class") == "rate_limited")
    fallbacks = sum(1 for item in events if item.get("used_fallback") or item.get("status") == "fallback")
    deadlines = sum(1 for item in events if item.get("error_class") == "deadline_exceeded")
    total_latency = sum(float(item.get("latency_ms") or 0) for item in events)
    ratio = (
        float(physical_attempts) / float(logical_calls)
        if logical_calls > 0
        else 0.0
    )
    return {
        "logical_provider_calls": logical_calls,
        "physical_provider_attempts": physical_attempts,
        "retry_amplification_ratio": round(ratio, 4),
        "successes": successes,
        "retries": retries,
        "timeouts": timeouts,
        "rate_limits": rate_limits,
        "fallbacks": fallbacks,
        "deadline_exceeded": deadlines,
        "total_provider_latency_ms": round(total_latency, 2),
    }


def call_with_policy(
    fn: Callable[[], T],
    *,
    policy: ProviderCallPolicy,
    context: ProviderCallContext | None = None,
    is_retryable: Callable[[BaseException], bool] | None = None,
) -> T:
    """Single retry owner for one logical provider call."""
    ctx = context or ProviderCallContext.create()
    logical_call_id = uuid.uuid4().hex
    retryable = is_retryable or is_retryable_provider_exception
    attempts = max(1, int(policy.max_attempts))
    last_error: BaseException | None = None

    for attempt in range(attempts):
        remaining = ctx.ensure_budget()
        started = ctx.now()
        status = "error"
        error_class: str | None = None
        status_code: int | None = None
        retry_after: float | None = None
        backoff_ms = 0.0
        try:
            result = fn()
            status = "success"
            _append_trace(
                ctx,
                policy=policy,
                logical_call_id=logical_call_id,
                attempt=attempt + 1,
                status=status,
                status_code=status_code,
                error_class=None,
                latency_ms=(ctx.now() - started) * 1000.0,
                backoff_ms=0.0,
                retry_after_ms=None,
                used_fallback=False,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            error_class = classify_provider_exception(exc)
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = int(exc.response.status_code)
                retry_after = extract_retry_after_seconds(exc)
            latency_ms = (ctx.now() - started) * 1000.0
            can_retry = retryable(exc) and attempt < attempts - 1
            if not can_retry:
                _append_trace(
                    ctx,
                    policy=policy,
                    logical_call_id=logical_call_id,
                    attempt=attempt + 1,
                    status="error",
                    status_code=status_code,
                    error_class=error_class,
                    latency_ms=latency_ms,
                    backoff_ms=0.0,
                    retry_after_ms=(retry_after * 1000.0) if retry_after is not None else None,
                    used_fallback=False,
                    message=redact_provider_message(str(exc)),
                )
                raise

            try:
                remaining_after = ctx.ensure_budget()
            except DeadlineExceededError:
                _append_trace(
                    ctx,
                    policy=policy,
                    logical_call_id=logical_call_id,
                    attempt=attempt + 1,
                    status="error",
                    status_code=status_code,
                    error_class="deadline_exceeded",
                    latency_ms=latency_ms,
                    backoff_ms=0.0,
                    retry_after_ms=(retry_after * 1000.0) if retry_after is not None else None,
                    used_fallback=False,
                    message=redact_provider_message(str(exc)),
                )
                raise DeadlineExceededError(
                    f"request deadline exceeded before retry of {policy.provider}/{policy.operation}"
                ) from exc

            delay = compute_backoff_seconds(
                attempt_index=attempt,
                policy=policy,
                retry_after_seconds=retry_after,
                remaining_seconds=remaining_after,
                rng=ctx.rng,
            )
            if delay <= 0 and remaining_after is not None and remaining_after < 0.05:
                raise DeadlineExceededError(
                    f"request deadline exceeded before backoff for {policy.provider}/{policy.operation}"
                ) from exc
            backoff_ms = delay * 1000.0
            _append_trace(
                ctx,
                policy=policy,
                logical_call_id=logical_call_id,
                attempt=attempt + 1,
                status="retry",
                status_code=status_code,
                error_class=error_class,
                latency_ms=latency_ms,
                backoff_ms=backoff_ms,
                retry_after_ms=(retry_after * 1000.0) if retry_after is not None else None,
                used_fallback=False,
                message=redact_provider_message(str(exc)),
            )
            if delay > 0:
                ctx.sleep(delay)

    assert last_error is not None
    raise last_error


def _append_trace(
    ctx: ProviderCallContext,
    *,
    policy: ProviderCallPolicy,
    logical_call_id: str,
    attempt: int,
    status: str,
    status_code: int | None,
    error_class: str | None,
    latency_ms: float,
    backoff_ms: float,
    retry_after_ms: float | None,
    used_fallback: bool,
    message: str | None = None,
) -> None:
    if ctx.trace_sink is None:
        return
    remaining = ctx.remaining_seconds()
    event = {
        "request_id": ctx.request_id,
        "thread_id": ctx.thread_id,
        "provider": policy.provider,
        "operation": policy.operation,
        "logical_call_id": logical_call_id,
        "attempt": attempt,
        "max_attempts": policy.max_attempts,
        "status": status,
        "status_code": status_code,
        "error_class": error_class,
        "latency_ms": round(latency_ms, 2),
        "backoff_ms": round(backoff_ms, 2),
        "retry_after_ms": round(retry_after_ms, 2) if retry_after_ms is not None else None,
        "deadline_remaining_ms": (
            None if remaining is None else round(max(0.0, remaining) * 1000.0, 2)
        ),
        "used_fallback": used_fallback,
    }
    if message:
        event["message"] = message
    ctx.trace_sink.append(event)


# ---------------------------------------------------------------------------
# Process-local HTTP client reuse (not shared across OS processes)
# ---------------------------------------------------------------------------

_CLIENTS: dict[str, httpx.Client] = {}
_CLIENT_INSTANCE_IDS: dict[str, str] = {}
_CLIENT_LOCK = None


def _client_lock():
    global _CLIENT_LOCK
    import threading

    if _CLIENT_LOCK is None:
        _CLIENT_LOCK = threading.Lock()
    return _CLIENT_LOCK


def get_shared_http_client(
    key: str,
    *,
    timeout: httpx.Timeout | float | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    with _client_lock():
        client = _CLIENTS.get(key)
        if client is not None and not client.is_closed:
            return client
        # trust_env=False avoids corporate HTTP(S)_PROXY hijacking localhost stub/tests.
        client = httpx.Client(timeout=timeout, headers=headers, trust_env=False)
        _CLIENTS[key] = client
        _CLIENT_INSTANCE_IDS[key] = uuid.uuid4().hex
        return client


def get_shared_http_client_instance_id(key: str) -> str | None:
    """Stable process-local transport identity (not a Python object address)."""
    with _client_lock():
        return _CLIENT_INSTANCE_IDS.get(key)


def close_shared_http_clients() -> None:
    import logging

    logger = logging.getLogger(__name__)
    with _client_lock():
        for key, client in list(_CLIENTS.items()):
            try:
                if not client.is_closed:
                    client.close()
                    logger.info("closed shared HTTP client key=%s", key)
            finally:
                _CLIENTS.pop(key, None)
                _CLIENT_INSTANCE_IDS.pop(key, None)
    # Always emit a stable marker for Docker lifespan assertions.
    msg = "shared HTTP clients closed; provider transport cleanup completed"
    logger.info(msg)
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# Per-process bulkheads
# ---------------------------------------------------------------------------

_SEMAPHORES: dict[str, Any] = {}
_INFLIGHT: dict[str, int] = {}
_MAX_INFLIGHT_SEEN: dict[str, int] = {}
_INFLIGHT_LOCK = None


def _inflight_lock():
    global _INFLIGHT_LOCK
    import threading

    if _INFLIGHT_LOCK is None:
        _INFLIGHT_LOCK = threading.Lock()
    return _INFLIGHT_LOCK


def get_provider_semaphore(name: str, *, max_inflight: int):
    import threading

    key = f"{name}:{max_inflight}"
    sem = _SEMAPHORES.get(key)
    if sem is None:
        sem = threading.Semaphore(max(1, int(max_inflight)))
        _SEMAPHORES[key] = sem
    return sem


def provider_bulkhead_snapshot(name: str | None = None) -> dict[str, Any]:
    with _inflight_lock():
        if name is None:
            return {
                "inflight": dict(_INFLIGHT),
                "max_inflight_seen": dict(_MAX_INFLIGHT_SEEN),
            }
        return {
            "name": name,
            "inflight": int(_INFLIGHT.get(name, 0)),
            "max_inflight_seen": int(_MAX_INFLIGHT_SEEN.get(name, 0)),
        }


def acquire_provider_slot(
    name: str,
    *,
    max_inflight: int,
    context: ProviderCallContext,
    acquire_timeout_seconds: float = 5.0,
) -> Callable[[], None]:
    """Acquire a per-process slot; returns a release callback."""
    sem = get_provider_semaphore(name, max_inflight=max_inflight)
    remaining = context.remaining_seconds()
    timeout = float(acquire_timeout_seconds)
    if remaining is not None:
        timeout = min(timeout, max(0.0, remaining))
    if timeout <= 0:
        raise DeadlineExceededError(f"no budget left to acquire {name} bulkhead")
    if not sem.acquire(timeout=timeout):
        if context.remaining_seconds() is not None and (context.remaining_seconds() or 0) <= 0:
            raise DeadlineExceededError(f"deadline exceeded waiting for {name} bulkhead")
        raise ProviderBusyError(f"provider bulkhead busy: {name}")

    with _inflight_lock():
        current = int(_INFLIGHT.get(name, 0)) + 1
        _INFLIGHT[name] = current
        _MAX_INFLIGHT_SEEN[name] = max(int(_MAX_INFLIGHT_SEEN.get(name, 0)), current)

    def _release() -> None:
        with _inflight_lock():
            _INFLIGHT[name] = max(0, int(_INFLIGHT.get(name, 0)) - 1)
        sem.release()

    return _release
