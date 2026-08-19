"""Read-only adapter and deterministic split manifest for LEDGER."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .governance import HoldoutError

LEDGER_DATASET_ID = "artefactory/ledger-long-context-KPI-QA"
LEDGER_SOURCE_CONFIG = "eval"
LEDGER_SOURCE_SPLIT = "test"
LEDGER_DATA_LICENSE = "CC-BY-4.0"
LEDGER_CODE_LICENSE = "MIT"
LEDGER_PAGE_DELIMITER = "<--- Page Split --->"
PUBLIC_DEV = "public_dev"
PUBLIC_HOLDOUT = "public_holdout"
SPLIT_ALGORITHM = "sha256_company_key_v1"
MANIFEST_SCHEMA = "lumenfin_public_benchmark_manifest.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_PAGE_SPLIT_RE = re.compile(re.escape(LEDGER_PAGE_DELIMITER), re.IGNORECASE)


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise HoldoutError(f"LEDGER row needs non-empty {field}")
    return value.strip()


def _optional_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if value is None:
        return ""
    return str(value).strip()


def ledger_company_key(row: Mapping[str, Any]) -> str:
    exchange = _required_text(row, "exchange").casefold()
    ticker = _required_text(row, "ticker").casefold()
    return f"{exchange}:{ticker}"


def _normalize_qrels(raw_qrels: object) -> tuple[dict[str, int], ...]:
    if isinstance(raw_qrels, (str, bytes, Mapping)) or not isinstance(
        raw_qrels, Iterable
    ):
        raise HoldoutError("LEDGER qrels must be a sequence")
    normalized: list[dict[str, int]] = []
    seen: set[str] = set()
    for item in raw_qrels:
        if not isinstance(item, Mapping):
            raise HoldoutError("LEDGER qrels entries must be objects")
        doc_id = _required_text(item, "doc_id")
        if doc_id in seen:
            raise HoldoutError("LEDGER qrels contain duplicate doc_id")
        seen.add(doc_id)
        raw_relevance = item.get("relevance")
        if isinstance(raw_relevance, bool):
            raise HoldoutError("LEDGER qrel relevance must be 0, 1, or 2")
        try:
            relevance = int(raw_relevance)
        except (TypeError, ValueError) as exc:
            raise HoldoutError("LEDGER qrel relevance must be 0, 1, or 2") from exc
        if raw_relevance != relevance or relevance not in {0, 1, 2}:
            raise HoldoutError("LEDGER qrel relevance must be 0, 1, or 2")
        normalized.append({"doc_id": doc_id, "relevance": relevance})
    if not normalized:
        raise HoldoutError("LEDGER row needs non-empty qrels")
    if not any(item["relevance"] > 0 for item in normalized):
        raise HoldoutError("LEDGER row needs at least one positive qrel")
    return tuple(normalized)


def adapt_ledger_row(row: Mapping[str, Any]) -> dict[str, Any]:
    query_id = _required_text(row, "query_id")
    query_text = _required_text(row, "query_text")
    ticker = _required_text(row, "ticker")
    exchange = _required_text(row, "exchange")
    company_name = _required_text(row, "company_name")
    industry = _required_text(row, "industry")
    kpi = _required_text(row, "kpi")
    source = _required_text(row, "source")
    tag = _optional_text(row, "tag")
    raw_year = row.get("year")
    if isinstance(raw_year, bool):
        raise HoldoutError("LEDGER year must be an integer")
    try:
        year = int(raw_year)
    except (TypeError, ValueError) as exc:
        raise HoldoutError("LEDGER year must be an integer") from exc
    if raw_year != year or year < 1900 or year > 2100:
        raise HoldoutError("LEDGER year must be an integer in [1900, 2100]")

    mmd_text = row.get("mmd_text")
    if not isinstance(mmd_text, str) or not mmd_text.strip():
        raise HoldoutError("LEDGER row needs non-empty mmd_text")
    page_count = len(_PAGE_SPLIT_RE.split(mmd_text))
    if page_count < 2:
        raise HoldoutError("LEDGER mmd_text must contain page delimiters")
    qrels = _normalize_qrels(row.get("qrels"))
    expected_doc_prefix = f"{exchange}_{ticker}_".casefold()
    if any(
        re.fullmatch(
            re.escape(expected_doc_prefix) + r"\d{4}/page_\d{4,}",
            str(item["doc_id"]).casefold(),
        )
        is None
        for item in qrels
    ):
        raise HoldoutError(
            "LEDGER qrel doc_id is malformed or crosses the query exchange+ticker boundary"
        )

    return {
        "query_id": query_id,
        "query_text": query_text,
        "company_key": ledger_company_key(row),
        "ticker": ticker,
        "exchange": exchange,
        "company_name": company_name,
        "industry": industry,
        "year": year,
        "kpi": kpi,
        "source": source,
        "tag": tag,
        "qrels": qrels,
        "positive_qrels": sum(item["relevance"] > 0 for item in qrels),
        "primary_qrels": sum(item["relevance"] == 2 for item in qrels),
        "page_count": page_count,
    }


def assign_ledger_company_split(
    company_key: str,
    *,
    salt: str,
    holdout_fraction: float = 0.2,
) -> str:
    normalized_company = str(company_key or "").strip().casefold()
    normalized_salt = str(salt or "").strip()
    if not normalized_company:
        raise ValueError("company_key is required")
    if not normalized_salt:
        raise ValueError("split salt is required")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")
    digest = hashlib.sha256(
        f"{normalized_salt}\0{normalized_company}".encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(2**64)
    return PUBLIC_HOLDOUT if bucket < holdout_fraction else PUBLIC_DEV


def _ids_sha256(values: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_ledger_split_manifest(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_revision: str,
    source_artifact_sha256: str,
    salt: str,
    holdout_fraction: float = 0.2,
) -> dict[str, Any]:
    revision = str(source_revision or "").strip().lower()
    artifact_hash = str(source_artifact_sha256 or "").strip().lower()
    normalized_salt = str(salt or "").strip()
    if not _REVISION_RE.fullmatch(revision):
        raise HoldoutError("LEDGER source revision must be a pinned 40-character commit")
    if not _SHA256_RE.fullmatch(artifact_hash):
        raise HoldoutError("LEDGER source artifact SHA256 is invalid")
    if not normalized_salt:
        raise HoldoutError("LEDGER split salt is required")

    row_count = 0
    query_ids: set[str] = set()
    split_query_ids = {PUBLIC_DEV: [], PUBLIC_HOLDOUT: []}
    split_companies = {PUBLIC_DEV: set(), PUBLIC_HOLDOUT: set()}
    for raw_row in rows:
        record = adapt_ledger_row(raw_row)
        query_id = str(record["query_id"])
        if query_id in query_ids:
            raise HoldoutError("LEDGER snapshot contains duplicate query_id")
        query_ids.add(query_id)
        role = assign_ledger_company_split(
            str(record["company_key"]),
            salt=normalized_salt,
            holdout_fraction=holdout_fraction,
        )
        row_count += 1
        split_query_ids[role].append(query_id)
        split_companies[role].add(str(record["company_key"]))

    if row_count == 0:
        raise HoldoutError("LEDGER snapshot is empty")
    if not split_query_ids[PUBLIC_DEV] or not split_query_ids[PUBLIC_HOLDOUT]:
        raise HoldoutError("LEDGER company split must produce non-empty dev and holdout")
    if split_companies[PUBLIC_DEV] & split_companies[PUBLIC_HOLDOUT]:
        raise HoldoutError("LEDGER company split leaked companies across roles")

    snapshot_identity = {
        "dataset_id": LEDGER_DATASET_ID,
        "source_config": LEDGER_SOURCE_CONFIG,
        "source_split": LEDGER_SOURCE_SPLIT,
        "source_revision": revision,
        "source_artifact_sha256": artifact_hash,
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(snapshot_identity, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA,
        "dataset": snapshot_identity,
        "dataset_snapshot_sha256": snapshot_hash,
        "data_license": LEDGER_DATA_LICENSE,
        "code_license": LEDGER_CODE_LICENSE,
        "public_benchmark": True,
        "held_out_scope": "public_holdout_split_only",
        "foundation_model_training_exposure": "unknown",
        "held_out_claim": "public_company_disjoint_only",
        "product_accuracy_claim": False,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_salt_sha256": hashlib.sha256(
            normalized_salt.encode("utf-8")
        ).hexdigest(),
        "holdout_fraction": holdout_fraction,
        "rows": row_count,
        "query_ids_sha256": _ids_sha256(query_ids),
        "splits": {
            role: {
                "queries": len(split_query_ids[role]),
                "companies": len(split_companies[role]),
                "query_ids_sha256": _ids_sha256(split_query_ids[role]),
                "company_keys_sha256": _ids_sha256(split_companies[role]),
                "consumed": False,
                "local_tuning_allowed": role == PUBLIC_DEV,
                "held_out_from_local_tuning": role == PUBLIC_HOLDOUT,
            }
            for role in (PUBLIC_DEV, PUBLIC_HOLDOUT)
        },
        "scoring_enabled": False,
        "remote_calls": 0,
    }


def _ledger_parquet_files(path: str | Path) -> list[Path]:
    target = Path(path)
    files = sorted(target.rglob("*.parquet")) if target.is_dir() else [target]
    if not files or any(not item.is_file() for item in files):
        raise HoldoutError(f"LEDGER parquet snapshot not found: {target}")
    return files


def ledger_snapshot_sha256(path: str | Path) -> str:
    target = Path(path)
    files = _ledger_parquet_files(target)
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(target) if target.is_dir() else Path(file_path.name)
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def iter_ledger_parquet_rows(path: str | Path) -> Iterable[dict[str, Any]]:
    files = _ledger_parquet_files(path)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise HoldoutError(
            "pyarrow is required to read a local LEDGER parquet snapshot"
        ) from exc
    for file_path in files:
        try:
            parquet_file = parquet.ParquetFile(file_path)
            for batch in parquet_file.iter_batches(batch_size=64):
                yield from batch.to_pylist()
        except Exception as exc:
            raise HoldoutError(
                f"cannot read LEDGER parquet snapshot: {file_path.name}"
            ) from exc
