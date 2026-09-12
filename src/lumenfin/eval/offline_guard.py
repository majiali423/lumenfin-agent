"""Process-only offline overlay and outbound block for document-task eval.

Never writes ``.env``. Product ``AppConfig.from_env`` defaults are unchanged.
Import this module before building eval runners when ``--offline`` is set.
"""

from __future__ import annotations

import ipaddress
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

# Empty strings occupy process env so project dotenv cannot refill secrets.
EVAL_OFFLINE_PROCESS_ENV: dict[str, str] = {
    "APP_ENV": "test",
    "DATA_MODE": "live",
    "ALLOW_LOCAL_FALLBACK": "true",
    "DEEPSEEK_API_KEY": "",
    "DASHSCOPE_API_KEY": "",
    "ALPHAVANTAGE_API_KEY": "",
    "LANGSMITH_API_KEY": "",
    "LANGCHAIN_API_KEY": "",
    "MAS_LANGSMITH_TRACING": "false",
    "LANGCHAIN_TRACING_V2": "",
    "LANGSMITH_TRACING": "false",
    "MAS_EMBEDDING_PROVIDER": "deterministic",
    "MAS_EMBEDDING_DIMENSION": "384",
    "MAS_RAG_RERANK_PROVIDER": "lexical",
    "DASHSCOPE_RERANK_BASE_URL": "",
    "MAS_FETCH_LIVE_FUNDAMENTALS": "false",
    "MAS_FETCH_SEC_FUNDAMENTALS": "false",
    "MAS_BOUNDED_REPAIR": "false",
}

_BLOCK_STATE: dict[str, Any] = {
    "installed": False,
    "attempts": [],
    "orig_create_connection": None,
    "orig_getaddrinfo": None,
}


class OfflineNetworkError(RuntimeError):
    """Raised when offline eval tries to open a non-loopback socket."""


def apply_eval_offline_process_env(
    environ: dict[str, str] | os._Environ[str] | None = None,
) -> dict[str, str]:
    """Overwrite process env for this eval process only. Does not touch ``.env``."""
    target = os.environ if environ is None else environ
    for key, value in EVAL_OFFLINE_PROCESS_ENV.items():
        target[key] = value
    return {key: ("<empty>" if not value else "<set>") for key, value in EVAL_OFFLINE_PROCESS_ENV.items()}


def _host_allowed(host: str) -> bool:
    name = (host or "").strip().strip("[]").lower()
    if not name:
        return False
    if name in {"localhost", "127.0.0.1", "::1"}:
        return True
    try:
        ip = ipaddress.ip_address(name)
    except ValueError:
        return False
    return bool(ip.is_loopback)


def _record_and_maybe_raise(host: str) -> None:
    attempts: list[str] = _BLOCK_STATE["attempts"]
    attempts.append(host)
    if not _host_allowed(host):
        raise OfflineNetworkError(
            f"offline eval blocked outbound host {host!r}; "
            "lexical/deterministic overlay is not sufficient without this guard"
        )


def install_outbound_block() -> None:
    if _BLOCK_STATE["installed"]:
        return
    _BLOCK_STATE["orig_create_connection"] = socket.create_connection
    _BLOCK_STATE["orig_getaddrinfo"] = socket.getaddrinfo
    orig_create = socket.create_connection
    orig_getaddrinfo = socket.getaddrinfo

    def guarded_create_connection(address, *args, **kwargs):  # type: ignore[no-untyped-def]
        host = address[0] if isinstance(address, tuple) else str(address)
        _record_and_maybe_raise(str(host))
        return orig_create(address, *args, **kwargs)

    def guarded_getaddrinfo(host, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        if host is not None:
            _record_and_maybe_raise(str(host))
        return orig_getaddrinfo(host, port, *args, **kwargs)

    socket.create_connection = guarded_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
    _BLOCK_STATE["installed"] = True


def uninstall_outbound_block() -> None:
    if not _BLOCK_STATE["installed"]:
        return
    if _BLOCK_STATE["orig_create_connection"] is not None:
        socket.create_connection = _BLOCK_STATE["orig_create_connection"]
    if _BLOCK_STATE["orig_getaddrinfo"] is not None:
        socket.getaddrinfo = _BLOCK_STATE["orig_getaddrinfo"]
    _BLOCK_STATE["installed"] = False
    _BLOCK_STATE["orig_create_connection"] = None
    _BLOCK_STATE["orig_getaddrinfo"] = None


def blocked_attempts() -> list[str]:
    return list(_BLOCK_STATE["attempts"])


def reset_blocked_attempts() -> None:
    _BLOCK_STATE["attempts"] = []


@contextmanager
def outbound_block() -> Iterator[None]:
    install_outbound_block()
    try:
        yield
    finally:
        uninstall_outbound_block()
