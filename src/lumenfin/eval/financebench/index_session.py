"""Historical FinanceBench index session/tenant binding (eval-only).

The company-scope Milvus index was written by ``runner.py`` with
``session_id=tenant_id=financebench-eval``. Querying it with a different
session filter returns zero rows.
"""

from __future__ import annotations

from typing import Any

from ...rag.milvus_client import build_vector_filter_expr
from .index_inspect import (
    EXPECTED_CHUNKS,
    EXPECTED_COLLECTION,
    SOURCE_INDEX_SESSION_ID,
    SOURCE_INDEX_TENANT_ID,
)

FORBIDDEN_QUERY_SESSION_ID = "financebench-candidate-depth"
PREVIOUS_ATTEMPT_STATUS = "INVALID_SESSION_FILTER"
FIRST_ATTEMPT_OUTPUT_DIRNAME = "financebench_candidate_depth_test100"
LOCKED_OUTPUT_DIRNAME = "financebench_candidate_depth_test100_v2"
FAILED_PREFLIGHT_OUTPUT_DIRNAME = "financebench_candidate_depth_test100_v2_preflight"
LOCKED_PREFLIGHT_OUTPUT_DIRNAME = "financebench_candidate_depth_test100_v2_preflight2"
EMPTY_RETRIEVAL_FAIL_FAST = 3
DEFAULT_CANARY_COMPANIES = ("3M", "Apple", "Microsoft")
_SAMPLE_OUTPUT_FIELDS = (
    "session_id",
    "tenant_id",
    "companies",
    "primary_company",
    "document_id",
)


class IndexSessionError(ValueError):
    """Raised when the copied index cannot be queried with the source session."""


def resolve_source_scope(selected: dict[str, Any] | None) -> tuple[str, str]:
    payload = selected or {}
    session = str(payload.get("source_index_session_id") or SOURCE_INDEX_SESSION_ID).strip()
    tenant = str(payload.get("source_index_tenant_id") or SOURCE_INDEX_TENANT_ID).strip()
    if not session or not tenant:
        raise IndexSessionError("source index session_id/tenant_id is missing")
    return session, tenant


def resolve_query_session_id(requested: str | None, source_session_id: str) -> str:
    source = str(source_session_id or "").strip()
    if not source:
        raise IndexSessionError("source index session_id is missing")
    raw = str(requested or "").strip()
    if raw == FORBIDDEN_QUERY_SESSION_ID:
        raise IndexSessionError(
            "refusing query session_id 'financebench-candidate-depth'; "
            f"historical index was written with {SOURCE_INDEX_SESSION_ID}"
        )
    if raw and raw != source:
        raise IndexSessionError(
            f"query session_id {raw!r} does not match source index session {source!r}"
        )
    return source


def _live_row_count(client: Any, collection_name: str) -> int:
    try:
        stats = client.get_collection_stats(collection_name) or {}
    except Exception as exc:
        raise IndexSessionError(f"unable to read live row count: {exc}") from exc
    if not isinstance(stats, dict):
        raise IndexSessionError("unable to read live row count: stats are not a mapping")
    raw = stats.get("row_count", stats.get("rowCount"))
    if raw is None and isinstance(stats.get("stats"), dict):
        raw = stats["stats"].get("row_count")
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        raise IndexSessionError(f"unable to read live row count from {stats!r}") from exc


def _query_rows(
    client: Any,
    *,
    collection_name: str,
    filter_expr: str,
    limit: int = 1,
) -> list[dict[str, Any]]:
    rows = client.query(
        collection_name=collection_name,
        filter=filter_expr,
        output_fields=list(_SAMPLE_OUTPUT_FIELDS),
        limit=limit,
    )
    return list(rows or [])


def _load_copied_collection(client: Any, collection_name: str) -> None:
    try:
        loader = getattr(client, "load_collection", None)
        if not callable(loader):
            raise IndexSessionError(
                f"copied index client cannot load collection {collection_name}"
            )
        loader(collection_name)
    except IndexSessionError:
        raise
    except Exception as exc:
        raise IndexSessionError(
            f"failed to load copied collection {collection_name}: {exc}"
        ) from exc


def _best_effort_release(client: Any, collection_name: str) -> bool:
    releaser = getattr(client, "release_collection", None)
    if not callable(releaser):
        return False
    try:
        releaser(collection_name)
        return True
    except Exception:
        return False


def _best_effort_close(client: Any) -> None:
    closer = getattr(client, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:
        return


def verify_copied_index_scope(
    *,
    uri: str,
    collection_name: str = EXPECTED_COLLECTION,
    expected_session_id: str = SOURCE_INDEX_SESSION_ID,
    expected_tenant_id: str = SOURCE_INDEX_TENANT_ID,
    expected_row_count: int = EXPECTED_CHUNKS,
    canary_companies: tuple[str, ...] = DEFAULT_CANARY_COMPANIES,
    client: Any | None = None,
) -> dict[str, Any]:
    """Canary against the copied Milvus index.

    Loads the copied collection before query/search/get. Does not embed
    queries. Operates only on the copied index and does not modify the
    source index. Load/release change copied-collection runtime state.
    """
    owns_client = False
    work = client
    loaded = False
    released = False
    result: dict[str, Any] | None = None
    if work is None:
        from pymilvus import MilvusClient

        work = MilvusClient(uri)
        owns_client = True
    try:
        names = [str(item) for item in (work.list_collections() or [])]
        if collection_name not in names:
            raise IndexSessionError(f"copied index is missing collection {collection_name}")
        _load_copied_collection(work, collection_name)
        loaded = True
        row_count = _live_row_count(work, collection_name)
        if row_count <= 0:
            raise IndexSessionError("copied index live row count is 0")
        if int(expected_row_count) and row_count != int(expected_row_count):
            raise IndexSessionError(
                f"copied index live row count {row_count} != expected {expected_row_count}"
            )
        sample = _query_rows(work, collection_name=collection_name, filter_expr="id >= 0")
        if not sample:
            raise IndexSessionError("copied index sample query returned no rows")
        live_session = str(sample[0].get("session_id") or "").strip()
        live_tenant = str(sample[0].get("tenant_id") or "").strip()
        if live_session != expected_session_id:
            raise IndexSessionError(
                f"sample session_id {live_session!r} != expected {expected_session_id!r}"
            )
        if live_tenant != expected_tenant_id:
            raise IndexSessionError(
                f"sample tenant_id {live_tenant!r} != expected {expected_tenant_id!r}"
            )
        session_hits = _query_rows(
            work,
            collection_name=collection_name,
            filter_expr=build_vector_filter_expr(session_id=expected_session_id),
        )
        if not session_hits:
            raise IndexSessionError(
                f"session_id filter {expected_session_id!r} returned no rows"
            )
        tenant_hits = _query_rows(
            work,
            collection_name=collection_name,
            filter_expr=build_vector_filter_expr(tenant_id=expected_tenant_id),
        )
        if not tenant_hits:
            raise IndexSessionError(
                f"tenant_id filter {expected_tenant_id!r} returned no rows"
            )
        canary_company = ""
        for company in canary_companies:
            hits = _query_rows(
                work,
                collection_name=collection_name,
                filter_expr=build_vector_filter_expr(
                    session_id=expected_session_id,
                    companies=[company],
                ),
            )
            if hits:
                canary_company = company
                break
        if not canary_company:
            raise IndexSessionError(
                "company metadata canary returned no rows for "
                + ", ".join(canary_companies)
            )
        result = {
            "uri": uri,
            "collection_name": collection_name,
            "row_count": row_count,
            "session_id": expected_session_id,
            "tenant_id": expected_tenant_id,
            "canary_company": canary_company,
            "opened_milvus_client": True,
            "modified_source_index": False,
            "operated_on_copied_index_only": True,
            "collection_loaded": True,
            "collection_released_after_check": False,
            "query_embedding_calls": 0,
        }
        return result
    finally:
        if loaded:
            released = _best_effort_release(work, collection_name)
        if result is not None:
            result["collection_loaded"] = True
            result["collection_released_after_check"] = released
        if owns_client:
            _best_effort_close(work)
