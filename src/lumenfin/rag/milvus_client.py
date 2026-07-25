"""Milvus URI helpers, client pooling, and filter expression builders."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from pymilvus import MilvusClient

_CLIENT_POOL: dict[str, MilvusClient] = {}
_POOL_LOCK = threading.Lock()


def is_milvus_server_uri(uri: str) -> bool:
    lowered = (uri or "").strip().lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("tcp://")
        or lowered.startswith("grpc://")
    )


def milvus_backend_kind(uri: str) -> str:
    if is_milvus_server_uri(uri):
        return "milvus-server"
    if str(uri).endswith(".db"):
        return "milvus-lite"
    return "milvus"


def resolve_milvus_uri(uri: str, *, isolate: bool | None = None) -> str:
    """Resolve Lite path isolation vs shared Server URI.

    - Server URIs (http/https/tcp/grpc) are never PID-isolated.
    - Lite `.db` files isolate per PID by default to avoid multi-process locks.
    - Set isolate=False or MAS_MILVUS_ISOLATE=false to share one Lite file
      (single-process / explicit shared-demo only).
    """
    raw = (uri or "").strip()
    if not raw:
        return raw
    if is_milvus_server_uri(raw):
        return raw
    path = Path(raw)
    if path.suffix != ".db":
        return raw
    if isolate is None:
        flag = os.getenv("MAS_MILVUS_ISOLATE", "true").strip().lower()
        isolate = flag not in {"0", "false", "no"}
    if not isolate:
        return raw
    return str(path.with_name(f"{path.stem}_p{os.getpid()}{path.suffix}"))


def escape_expr_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def quote_list(values: list[str]) -> str:
    return ", ".join(f'"{escape_expr_value(item)}"' for item in values)


def company_match_expr(companies: list[str]) -> str:
    """Build a filter for comma-separated `companies` varchar tags.

    Matches exact tag or tag at start/middle/end of the CSV string.
    """
    parts: list[str] = []
    for company in companies:
        name = (company or "").strip()
        if not name:
            continue
        esc = escape_expr_value(name)
        parts.append(
            "("
            f'companies == "{esc}" or '
            f'companies like "{esc},%" or '
            f'companies like "%,{esc}" or '
            f'companies like "%,{esc},%"'
            ")"
        )
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " or ".join(parts) + ")"


def build_vector_filter_expr(
    *,
    session_id: str | None = None,
    tenant_id: str | None = None,
    document_ids: list[str] | None = None,
    source_document_ids: list[str] | None = None,
    companies: list[str] | None = None,
) -> str:
    filter_parts: list[str] = []
    if tenant_id:
        filter_parts.append(f'tenant_id == "{escape_expr_value(tenant_id)}"')
    elif session_id:
        filter_parts.append(f'session_id == "{escape_expr_value(session_id)}"')
    if source_document_ids:
        filter_parts.append(f"source_document_id in [{quote_list(source_document_ids)}]")
    elif document_ids:
        filter_parts.append(f"document_id in [{quote_list(document_ids)}]")
    if companies:
        company_expr = company_match_expr(companies)
        if company_expr:
            filter_parts.append(company_expr)
    return " and ".join(filter_parts)


def get_shared_milvus_client(uri: str) -> MilvusClient:
    """Reuse one MilvusClient per server URI within the process (worker-safe pool).

    Lite file URIs are not pooled: each store keeps its own client to match
    existing single-process Lite semantics and simplify close().
    """
    if not is_milvus_server_uri(uri):
        return MilvusClient(uri)
    with _POOL_LOCK:
        client = _CLIENT_POOL.get(uri)
        if client is None:
            client = MilvusClient(uri)
            _CLIENT_POOL[uri] = client
        return client


def clear_milvus_client_pool() -> None:
    """Test helper: drop pooled server clients."""
    with _POOL_LOCK:
        clients = list(_CLIENT_POOL.values())
        _CLIENT_POOL.clear()
    for client in clients:
        try:
            client.close()
        except Exception:
            pass


def milvus_client_pool_stats() -> dict[str, Any]:
    with _POOL_LOCK:
        return {"pooled_uris": list(_CLIENT_POOL.keys()), "size": len(_CLIENT_POOL)}
