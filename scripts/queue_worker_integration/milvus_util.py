from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.milvus_client import build_vector_filter_expr, get_shared_milvus_client


def count_vectors(
    uri: str,
    collection: str,
    *,
    tenant_id: str | None = None,
    source_document_id: str | None = None,
) -> int:
    client = get_shared_milvus_client(uri)
    if not client.has_collection(collection):
        return 0
    try:
        client.load_collection(collection)
    except Exception:
        pass
    expr = build_vector_filter_expr(
        tenant_id=tenant_id,
        source_document_ids=[source_document_id] if source_document_id else None,
    )
    kwargs: dict[str, Any] = {
        "collection_name": collection,
        "output_fields": ["id"],
        "limit": 16384,
    }
    if expr:
        kwargs["filter"] = expr
    rows = client.query(**kwargs)
    return len(rows or [])


def drop_collection(uri: str, collection: str) -> None:
    client = get_shared_milvus_client(uri)
    if client.has_collection(collection):
        client.drop_collection(collection)
