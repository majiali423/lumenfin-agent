"""Read-only inspection of existing FinanceBench eval indexes.

Opens sidecar JSON and Milvus Lite ``schema.json`` / collection ``manifest.json``
on disk. Does not construct ``MilvusRAGStore``, embed queries, parse PDFs, or
write into the original index directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .reporting import sha256_file
from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DASHSCOPE_EMBEDDING_DIM,
)

EXPECTED_DATASET_HASH = "5e961c0aa84a5ed578bdc2cea4f2ef8e33aa6ffe9394fc6c2508b303bf10fdeb"
EXPECTED_DOCUMENTS = 84
EXPECTED_CHUNKS = 52518
EXPECTED_EMBEDDING_MODEL = "text-embedding-v4"
EXPECTED_EMBEDDING_DIMENSION = DEFAULT_DASHSCOPE_EMBEDDING_DIM
EXPECTED_CHUNK_SIZE = DEFAULT_CHUNK_CHARS
EXPECTED_CHUNK_OVERLAP = DEFAULT_CHUNK_OVERLAP
EXPECTED_COLLECTION = "financebench_eval"
SOURCE_INDEX_COMMIT = "5877be8555bd72f411225b809ed75454607618bd"
SOURCE_INDEX_WORKTREE_DIRTY = True
SOURCE_INDEX_CHUNKER = "pre_overlap_fix"
REQUIRED_SCHEMA_FIELDS = frozenset(
    {
        "document_id",
        "source_document_id",
        "filename",
        "page",
        "companies",
        "primary_company",
        "text",
        "vector",
        "sparse",
    }
)
SECTION_METADATA = "NOT_AVAILABLE"

HISTORICAL_OUTPUT_DIRNAMES = (
    "financebench_eval",
    "financebench_eval_company",
    "financebench_eval_confirmation",
)


class IndexIncompatibleError(RuntimeError):
    """Raised when a reusable FinanceBench index cannot be proven compatible."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def historical_eval_dirs(repo_root: Path) -> tuple[Path, ...]:
    outputs = Path(repo_root) / "outputs"
    return tuple(outputs / name for name in HISTORICAL_OUTPUT_DIRNAMES)


def is_historical_output_path(path: Path, *, repo_root: Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    for historical in historical_eval_dirs(repo_root):
        target = historical.resolve()
        if resolved == target or target in resolved.parents:
            return True
    return False


def _collection_dir(eval_db: Path) -> Path | None:
    collections = eval_db / "collections"
    if not collections.is_dir():
        return None
    preferred = collections / EXPECTED_COLLECTION
    if preferred.is_dir():
        return preferred
    children = [path for path in collections.iterdir() if path.is_dir()]
    return children[0] if len(children) == 1 else None


def _schema_fields(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for field in schema.get("fields") or []:
        if isinstance(field, dict) and field.get("name"):
            names.add(str(field["name"]))
    return names


def _vector_dim(schema: dict[str, Any]) -> int | None:
    for field in schema.get("fields") or []:
        if not isinstance(field, dict):
            continue
        if str(field.get("name") or "") != "vector":
            continue
        dim = field.get("dim")
        try:
            return int(dim)
        except (TypeError, ValueError):
            return None
    return None


def _sidecar_for_index(index_root: Path) -> dict[str, Any]:
    eval_root = index_root.parent
    environment: dict[str, Any] = {}
    for mode in ("hybrid", "dense", "bm25", "hybrid-qwen3"):
        payload = _read_json(eval_root / mode / "environment.json")
        if payload:
            environment = payload
            break
    if not environment:
        environment = _read_json(eval_root / "environment.json")
    results = _read_json(eval_root / "results.json")
    ingestion = results.get("ingestion") if isinstance(results.get("ingestion"), dict) else {}
    if not ingestion:
        for mode in ("hybrid", "dense", "bm25", "hybrid-qwen3"):
            mode_results = _read_json(eval_root / mode / "results.json")
            if isinstance(mode_results.get("ingestion"), dict):
                ingestion = mode_results["ingestion"]
                break
    manifest = _read_json(eval_root / "hybrid" / "manifest.json") or _read_json(
        eval_root / "manifest.json"
    )
    return {
        "eval_root": str(eval_root),
        "environment": environment,
        "ingestion": ingestion,
        "manifest": manifest,
        "results_status": results.get("status") or "",
        "index_scope": str(
            environment.get("index_scope")
            or results.get("index_scope")
            or ""
        ),
    }


def inspect_lite_index(eval_db: Path, *, sidecar: dict[str, Any] | None = None) -> dict[str, Any]:
    """File-level compatibility report for one Milvus Lite URI directory."""
    eval_db = Path(eval_db)
    collection_dir = _collection_dir(eval_db) if eval_db.is_dir() else None
    schema_path = collection_dir / "schema.json" if collection_dir else None
    manifest_path = collection_dir / "manifest.json" if collection_dir else None
    schema = _read_json(schema_path) if schema_path else {}
    collection_manifest = _read_json(manifest_path) if manifest_path else {}
    sidecar = sidecar or {}
    environment = sidecar.get("environment") if isinstance(sidecar.get("environment"), dict) else {}
    ingestion = sidecar.get("ingestion") if isinstance(sidecar.get("ingestion"), dict) else {}
    manifest = sidecar.get("manifest") if isinstance(sidecar.get("manifest"), dict) else {}
    fields = _schema_fields(schema)
    missing_fields = sorted(REQUIRED_SCHEMA_FIELDS - fields)
    index_specs = collection_manifest.get("index_specs") or {}
    has_vector_index = isinstance(index_specs.get("vector"), dict)
    has_sparse_index = isinstance(index_specs.get("sparse"), dict)
    row_seq = collection_manifest.get("current_seq")
    try:
        row_count = int(row_seq)
    except (TypeError, ValueError):
        row_count = int(ingestion.get("chunks_created") or 0)
    document_count = int(
        ingestion.get("document_count")
        or manifest.get("document_count")
        or collection_manifest.get("active_wal_number")
        or 0
    )
    dataset_hash = str(
        environment.get("dataset_hash") or manifest.get("dataset_hash") or ""
    )
    embedding_model = str(environment.get("embedding_model") or "")
    try:
        embedding_dimension = int(
            environment.get("embedding_dimension") or _vector_dim(schema) or 0
        )
    except (TypeError, ValueError):
        embedding_dimension = int(_vector_dim(schema) or 0)
    try:
        chunk_size = int(environment.get("chunk_size") or manifest.get("chunk_size") or 0)
    except (TypeError, ValueError):
        chunk_size = 0
    try:
        chunk_overlap = int(
            environment.get("chunk_overlap") or manifest.get("chunk_overlap") or 0
        )
    except (TypeError, ValueError):
        chunk_overlap = 0
    checks = {
        "dataset_hash": dataset_hash == EXPECTED_DATASET_HASH,
        "documents": document_count == EXPECTED_DOCUMENTS,
        "chunks": row_count == EXPECTED_CHUNKS,
        "embedding_model": embedding_model == EXPECTED_EMBEDDING_MODEL,
        "embedding_dimension": embedding_dimension == EXPECTED_EMBEDDING_DIMENSION,
        "chunk_size": chunk_size == EXPECTED_CHUNK_SIZE,
        "chunk_overlap": chunk_overlap == EXPECTED_CHUNK_OVERLAP,
        "schema_fields": not missing_fields,
        "vector_index": has_vector_index,
        "sparse_index": has_sparse_index,
        "collection_present": collection_dir is not None and bool(schema),
        "page_metadata": "page" in fields,
        "document_metadata": "document_id" in fields,
        "company_metadata": "companies" in fields or "primary_company" in fields,
    }
    compatible = all(checks.values())
    mismatches = [name for name, ok in checks.items() if not ok]
    return {
        "uri": str(eval_db),
        "index_root": str(eval_db.parent),
        "eval_root": sidecar.get("eval_root") or str(eval_db.parent.parent),
        "index_scope": sidecar.get("index_scope") or "",
        "collection_name": EXPECTED_COLLECTION,
        "compatible": compatible,
        "mismatches": mismatches,
        "checks": checks,
        "dataset_hash": dataset_hash,
        "documents": document_count,
        "chunks": row_count,
        "embedding_provider": str(environment.get("embedding_provider") or ""),
        "embedding_model": embedding_model,
        "embedding_dimension": embedding_dimension,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "bm25_rrf_weight": float(environment.get("bm25_rrf_weight") or DEFAULT_BM25_RRF_WEIGHT),
        "missing_schema_fields": missing_fields,
        "section_metadata": SECTION_METADATA,
        "zero_chunk_documents": list(ingestion.get("zero_chunk_documents") or []),
        "has_lock_file": (eval_db / "LOCK").exists(),
        "readonly_guaranteed": False,
        "copy_required": True,
        "schema_complete": not missing_fields and has_vector_index and has_sparse_index,
        "source_index_commit": SOURCE_INDEX_COMMIT,
        "source_index_worktree_dirty": SOURCE_INDEX_WORKTREE_DIRTY,
        "source_index_chunker": SOURCE_INDEX_CHUNKER,
        "source_schema_sha256": sha256_file(schema_path) if schema_path and schema_path.is_file() else "",
        "source_collection_manifest_sha256": (
            sha256_file(manifest_path) if manifest_path and manifest_path.is_file() else ""
        ),
        "index_not_current_chunker": True,
    }


def discover_eval_indexes(repo_root: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for eval_root in historical_eval_dirs(repo_root):
        if not eval_root.is_dir():
            continue
        for index_root in sorted(eval_root.glob("index-*")):
            eval_db = index_root / "eval.db"
            if not eval_db.exists():
                continue
            sidecar = _sidecar_for_index(index_root)
            reports.append(inspect_lite_index(eval_db, sidecar=sidecar))
    return reports


def select_compatible_index(reports: list[dict[str, Any]]) -> dict[str, Any] | None:
    compatible = [item for item in reports if item.get("compatible")]
    if not compatible:
        return None
    company = [item for item in compatible if item.get("index_scope") == "company"]
    pool = company or compatible
    pool.sort(key=lambda item: (0 if "financebench_eval_company" in str(item.get("eval_root") or "") else 1, str(item.get("uri") or "")))
    return pool[0]


def inspect_financebench_indexes(repo_root: Path | str) -> dict[str, Any]:
    root = Path(repo_root)
    reports = discover_eval_indexes(root)
    selected = select_compatible_index(reports)
    return {
        "expected": {
            "dataset_hash": EXPECTED_DATASET_HASH,
            "documents": EXPECTED_DOCUMENTS,
            "chunks": EXPECTED_CHUNKS,
            "embedding_model": EXPECTED_EMBEDDING_MODEL,
            "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "chunk_size": EXPECTED_CHUNK_SIZE,
            "chunk_overlap": EXPECTED_CHUNK_OVERLAP,
            "collection_name": EXPECTED_COLLECTION,
            "section_metadata": SECTION_METADATA,
        },
        "candidates": reports,
        "compatible_index": selected,
        "compatible": bool(selected),
        "reembed_chunks": False,
        "opened_milvus_client": False,
        "modified_original_index": False,
    }


def require_compatible_index(inspection: dict[str, Any]) -> dict[str, Any]:
    selected = inspection.get("compatible_index")
    if not isinstance(selected, dict) or not selected.get("compatible"):
        raise IndexIncompatibleError(
            "no compatible FinanceBench Milvus index found; refusing to re-embed "
            f"{EXPECTED_CHUNKS} chunks. Inspected dataset_hash must be "
            f"{EXPECTED_DATASET_HASH}, documents={EXPECTED_DOCUMENTS}, "
            f"chunks={EXPECTED_CHUNKS}, embedding={EXPECTED_EMBEDDING_MODEL}/"
            f"{EXPECTED_EMBEDDING_DIMENSION}, chunk={EXPECTED_CHUNK_SIZE}/"
            f"{EXPECTED_CHUNK_OVERLAP}, with company/document/page metadata "
            "and complete dense+BM25 indexes."
        )
    return selected
