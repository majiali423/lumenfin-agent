"""Bounded retries for transient external-provider failures."""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

import httpx

T = TypeVar("T")

TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
TRANSIENT_ERROR_CLASSES = frozenset(
    {"timeout", "connection", "rate_limited", "transient_http"}
)


def classify_http_status(status_code: int) -> str:
    if status_code == 404:
        return "not_found"
    if status_code == 429:
        return "rate_limited"
    if status_code in TRANSIENT_HTTP_STATUS:
        return "transient_http"
    if 400 <= status_code < 500:
        return "client_error"
    return "http_error"


def classify_exception(exc: BaseException) -> str:
    """Map provider exceptions to stable error classes for audit/fatal summaries."""
    msg = str(exc).lower()
    name = type(exc).__name__.lower()

    if isinstance(exc, httpx.HTTPStatusError):
        return classify_http_status(exc.response.status_code)

    if isinstance(exc, (TimeoutError, httpx.TimeoutException)) or "timeout" in name or "timed out" in msg:
        return "timeout"
    if isinstance(
        exc,
        (ConnectionError, httpx.ConnectError, httpx.NetworkError, httpx.RemoteProtocolError),
    ) or "connection" in name:
        return "connection"
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return "rate_limited"
    if "503" in msg or "502" in msg or "504" in msg:
        return "transient_http"
    return "error"


def is_transient_error_class(error_class: str) -> bool:
    return error_class in TRANSIENT_ERROR_CLASSES


def is_transient_exception(exc: BaseException) -> bool:
    return is_transient_error_class(classify_exception(exc))


def call_with_transient_retry(
    fn: Callable[[], T],
    *,
    max_retries: int = 3,
    backoff_seconds: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry only timeout / connection / 429 / 5xx-class failures."""
    last_error: BaseException | None = None
    attempts = max(1, max_retries)
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if not is_transient_exception(exc) or attempt >= attempts - 1:
                raise
            sleep(backoff_seconds * (2**attempt))
    assert last_error is not None
    raise last_error


def append_provider_error(
    sink: list[dict[str, Any]] | None,
    *,
    provider: str,
    symbol: str,
    error_class: str,
    message: str,
    attempts: int = 1,
) -> None:
    if sink is None:
        return
    sink.append(
        {
            "provider": provider,
            "symbol": symbol,
            "error_class": error_class,
            "message": message[:500],
            "attempts": attempts,
            "transient": is_transient_error_class(error_class),
        }
    )


def summarize_provider_errors(errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Split provider failures into transient vs truly-missing for fatal routing audit."""
    transient = [item for item in errors if item.get("transient")]
    missing = [
        item
        for item in errors
        if item.get("error_class") in {"not_found", "truly_missing", "unavailable"}
    ]
    other = [
        item
        for item in errors
        if item not in transient and item not in missing
    ]
    return {
        "count": len(errors),
        "transient_count": len(transient),
        "missing_count": len(missing),
        "other_count": len(other),
        "has_transient": bool(transient),
        "has_truly_missing": bool(missing) and not transient,
        "by_class": _count_by(errors, "error_class"),
        "by_provider": _count_by(errors, "provider"),
        "items": list(errors),
    }


def _count_by(errors: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in errors:
        label = str(item.get(key) or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts
