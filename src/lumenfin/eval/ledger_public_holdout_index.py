"""Document-only LEDGER public_holdout index. Never reads query/gold/qrels."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..rag.chunking import chunk_document
from .holdout.ledger import (
    LEDGER_PAGE_DELIMITER,
    PUBLIC_HOLDOUT,
    assign_ledger_company_split,
    ledger_company_key,
    ledger_snapshot_sha256,
)

PRODUCT_TAG = "v0.1.0-rc.5"
PRODUCT_COMMIT = "31e8680aa89636f1fd897d7aa5ed7ca86317bd73"
SPLIT_SALT = "lumenfin-ledger-public-v1"
SPLIT_SALT_SHA256 = "b1f8fb42797fb020821d42142bb74e70579688016f7d2c680a8b64502129fd2c"
DEFAULT_SNAPSHOT = (
    Path("data")
    / "external"
    / "ledger-long-context-KPI-QA"
    / "b7085dc6cb16b3ec8149a9baf6dd2d3416cf7619"
    / "eval-test"
)
MANIFEST_PATH = Path("data") / "eval_rag" / "holdout" / "ledger_public_manifest.json"
INDEX_DIR = Path("outputs") / "ledger_public_holdout_rc5_index_v1"
DRYRUN_DIR = Path("outputs") / "ledger_public_holdout_rc5_index_v1_dryrun"
SEAL_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_index_v1.json"
COLLECTION = "lumenfin_ledger_public_holdout_rc5_v1"
SESSION_ID = "ledger-public-holdout-rc5-index-v1"
ALLOWED_COLUMNS = ("exchange", "ticker", "company_name", "year", "mmd_text")
FORBIDDEN_COLUMNS = (
    "query_text",
    "value",
    "qrels",
    "query_id",
    "kpi",
    "gold",
    "expected",
    "industry",
    "source",
    "tag",
)
RC5_SOURCE_FILES = (
    Path("src") / "lumenfin" / "rag" / "chunking.py",
    Path("src") / "lumenfin" / "rag" / "embeddings.py",
    Path("src") / "lumenfin" / "rag" / "milvus_store.py",
    Path("src") / "lumenfin" / "rag" / "hybrid_retriever.py",
)
# China-mainland DashScope sync text-embedding-v4, not Batch and not intl.
PRICE_CNY_PER_MILLION = 0.5
PRICE_CNY_PER_1K = 0.0005
PRICE_SOURCE = "https://help.aliyun.com/zh/model-studio/text-embedding-v4"
PRICE_REGION = "dashscope.aliyuncs.com China mainland sync (Beijing / 华北2)"
DASHSCOPE_BATCH = 10
MAX_TOKENS = 50_000_000
MAX_HTTP = 10_000
MAX_FEE_CNY = 30.0
PAGE_SPLIT_RE = re.compile(re.escape(LEDGER_PAGE_DELIMITER), re.IGNORECASE)
CJK_RE = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
CANARY_QUERY = "operating income cash flow total assets revenue"


class HoldoutIndexError(ValueError):
    """Safe index-governance error. No query, gold, qrels, or secrets."""


@dataclass
class InputAccessAudit:
    allowed_fields: tuple[str, ...] = ALLOWED_COLUMNS
    requested_fields: tuple[str, ...] = ALLOWED_COLUMNS
    schema_fields: tuple[str, ...] = ()
    actual_read_fields: tuple[str, ...] = ()
    refused_fields: tuple[str, ...] = FORBIDDEN_COLUMNS
    projection_used: bool = False
    holdout_query_text_accessed: bool = False
    holdout_gold_accessed: bool = False
    holdout_qrels_accessed: bool = False
    corpus_document_text_accessed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed_fields": list(self.allowed_fields),
            "requested_fields": list(self.requested_fields),
            "schema_fields": list(self.schema_fields),
            "actual_read_fields": list(self.actual_read_fields),
            "refused_fields": list(self.refused_fields),
            "projection_used": self.projection_used,
            "corpus_document_text_accessed": self.corpus_document_text_accessed,
            "holdout_query_text_accessed": self.holdout_query_text_accessed,
            "holdout_gold_accessed": self.holdout_gold_accessed,
            "holdout_qrels_accessed": self.holdout_qrels_accessed,
            "holdout_consumed": False,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def ids_sha256(values: list[str] | tuple[str, ...] | set[str]) -> str:
    canonical = "\n".join(sorted(str(item) for item in values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def canonical_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def estimate_tokens(text: str) -> dict[str, int]:
    """Local token bounds. No remote tokenizer, no query/gold/qrels."""
    chars = len(text)
    cjk = len(CJK_RE.findall(text))
    other = max(0, chars - cjk)
    typical = cjk + math.ceil(other / 4) if chars else 0
    conservative = cjk + math.ceil(other / 2) if chars else 0
    return {
        "chars": chars,
        "typical_qwen_style": typical,
        "conservative_mixed": conservative,
        "hard_upper_char": chars,
    }


def assert_rc5_sources(*, repo_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in RC5_SOURCE_FILES:
        current = sha256_path(repo_root / rel)
        shown = subprocess.check_output(
            ["git", "show", f"{PRODUCT_COMMIT}:{rel.as_posix()}"],
            cwd=repo_root,
        )
        if current != sha256_bytes(shown):
            raise HoldoutIndexError(f"source {rel.as_posix()} differs from rc5")
        hashes[rel.as_posix()] = current
    if hashlib.sha256(SPLIT_SALT.encode("utf-8")).hexdigest() != SPLIT_SALT_SHA256:
        raise HoldoutIndexError("split salt does not match frozen manifest")
    return hashes


def assert_clean_projection(columns: tuple[str, ...]) -> None:
    forbidden = [item for item in columns if item in FORBIDDEN_COLUMNS]
    if forbidden:
        raise HoldoutIndexError("requested columns include forbidden holdout fields")


def parquet_schema_fields(snapshot: Path) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise HoldoutIndexError("pyarrow is required") from exc
    files = sorted(snapshot.rglob("*.parquet")) if snapshot.is_dir() else [snapshot]
    if not files:
        raise HoldoutIndexError("LEDGER parquet snapshot not found")
    return tuple(parquet.ParquetFile(files[0]).schema_arrow.names)


def iter_projected_corpus_rows(
    snapshot: Path,
    *,
    columns: tuple[str, ...] = ALLOWED_COLUMNS,
) -> Iterator[dict[str, Any]]:
    assert_clean_projection(columns)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise HoldoutIndexError("pyarrow is required") from exc
    files = sorted(snapshot.rglob("*.parquet")) if snapshot.is_dir() else [snapshot]
    if not files:
        raise HoldoutIndexError("LEDGER parquet snapshot not found")
    for file_path in files:
        handle = parquet.ParquetFile(file_path)
        schema = set(handle.schema_arrow.names)
        missing = [item for item in columns if item not in schema]
        if missing:
            raise HoldoutIndexError("snapshot is missing required corpus columns")
        if not schema.intersection(FORBIDDEN_COLUMNS):
            raise HoldoutIndexError(
                "cannot prove column isolation without forbidden fields in schema"
            )
        for batch in handle.iter_batches(batch_size=64, columns=list(columns)):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise HoldoutIndexError("projected row is not an object")
                if set(row) - set(columns):
                    raise HoldoutIndexError("projection leaked unrequested fields")
                if set(row).intersection(FORBIDDEN_COLUMNS):
                    raise HoldoutIndexError("projection leaked forbidden holdout fields")
                yield row


def page_documents(
    report_id: str,
    company_key: str,
    company_name: str,
    mmd_text: str,
) -> tuple[list[dict[str, Any]], int]:
    blank = 0
    documents: list[dict[str, Any]] = []
    for page_zero, raw_page in enumerate(PAGE_SPLIT_RE.split(mmd_text)):
        text = raw_page.strip()
        if not text:
            blank += 1
            continue
        ledger_doc_id = f"{report_id}/page_{page_zero:04d}"
        documents.append(
            {
                "document_id": ledger_doc_id,
                "source_document_id": ledger_doc_id,
                "filename": f"{report_id}.mmd",
                "pages": [text],
                "issuer_companies": [company_key],
                "detected_companies": [company_key],
                "ledger_doc_id": ledger_doc_id,
                "ledger_company_name": company_name,
                "ledger_page_zero": page_zero,
            }
        )
    return documents, blank


def scan_holdout_reports(
    snapshot: Path,
    *,
    salt: str = SPLIT_SALT,
) -> tuple[dict[str, tuple[str, str, str, str]], InputAccessAudit, int]:
    audit = InputAccessAudit(
        schema_fields=parquet_schema_fields(snapshot),
        actual_read_fields=ALLOWED_COLUMNS,
        projection_used=True,
        corpus_document_text_accessed=True,
    )
    reports: dict[str, tuple[str, str, str, str]] = {}
    rows_seen = 0
    for row in iter_projected_corpus_rows(snapshot):
        rows_seen += 1
        company_key = ledger_company_key(row)
        if assign_ledger_company_split(company_key, salt=salt) != PUBLIC_HOLDOUT:
            continue
        year = row.get("year")
        try:
            year_int = int(year)
        except (TypeError, ValueError) as exc:
            raise HoldoutIndexError("corpus year is not an integer") from exc
        report_id = f"{row['exchange']}_{row['ticker']}_{year_int}"
        mmd_text = str(row.get("mmd_text") or "")
        content_hash = hashlib.sha256(mmd_text.encode("utf-8")).hexdigest()
        identity = (content_hash, company_key, str(row.get("company_name") or ""))
        previous = reports.get(report_id)
        if previous is not None and previous[:3] != identity:
            raise HoldoutIndexError("repeated report has inconsistent content")
        reports.setdefault(report_id, (*identity, mmd_text))
    return reports, audit, rows_seen


def materialize_documents(
    reports: Mapping[str, tuple[str, str, str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    companies: set[str] = set()
    blank_pages = 0
    empty_reports = 0
    for report_id, (_content_hash, company_key, company_name, mmd_text) in sorted(
        reports.items()
    ):
        companies.add(company_key)
        pages, blank = page_documents(report_id, company_key, company_name, mmd_text)
        blank_pages += blank
        if not pages:
            empty_reports += 1
            continue
        documents.extend(pages)
    stats = {
        "holdout_companies": len(companies),
        "company_keys_sha256": ids_sha256(companies),
        "reports": len(reports),
        "documents": len(documents),
        "blank_pages": blank_pages,
        "empty_reports": empty_reports,
        "document_ids_sha256": ids_sha256(
            [str(item["document_id"]) for item in documents]
        ),
        "companies": sorted(companies),
    }
    return documents, stats


def verify_official_identity(
    *,
    repo_root: Path,
    snapshot: Path,
    stats: Mapping[str, Any],
) -> dict[str, str]:
    manifest = json.loads((repo_root / MANIFEST_PATH).read_text(encoding="utf-8"))
    expected_companies = str(manifest["splits"]["public_holdout"]["company_keys_sha256"])
    expected_count = int(manifest["splits"]["public_holdout"]["companies"])
    expected_artifact = str(manifest["dataset"]["source_artifact_sha256"])
    actual_artifact = ledger_snapshot_sha256(snapshot)
    if actual_artifact != expected_artifact:
        raise HoldoutIndexError("parquet snapshot hash does not match frozen manifest")
    if stats["company_keys_sha256"] != expected_companies:
        raise HoldoutIndexError("holdout company set does not match frozen manifest")
    if int(stats["holdout_companies"]) != expected_count:
        raise HoldoutIndexError("holdout company count does not match frozen manifest")
    return {
        "source_artifact_sha256": actual_artifact,
        "company_keys_sha256": expected_companies,
    }


def estimate_from_documents(documents: list[dict[str, Any]]) -> dict[str, Any]:
    chunk_count = 0
    char_count = 0
    typical_tokens = 0
    conservative_tokens = 0
    hard_tokens = 0
    oversized = 0
    for document in documents:
        for chunk in chunk_document(document):
            chunk_count += 1
            text = str(chunk.get("text") or "")
            bounds = estimate_tokens(text)
            char_count += bounds["chars"]
            typical_tokens += bounds["typical_qwen_style"]
            conservative_tokens += bounds["conservative_mixed"]
            hard_tokens += bounds["hard_upper_char"]
            if bounds["conservative_mixed"] > 8192:
                oversized += 1
    http_requests = math.ceil(chunk_count / DASHSCOPE_BATCH) if chunk_count else 0
    fee = (conservative_tokens / 1_000_000.0) * PRICE_CNY_PER_MILLION
    typical_fee = (typical_tokens / 1_000_000.0) * PRICE_CNY_PER_MILLION
    hard_fee = (hard_tokens / 1_000_000.0) * PRICE_CNY_PER_MILLION
    disk_mb = (chunk_count * 1024 * 4 + char_count) / (1024 * 1024)
    return {
        "chunks": chunk_count,
        "embed_chars": char_count,
        "estimated_tokens_typical_qwen_style": typical_tokens,
        "estimated_tokens_conservative_mixed": conservative_tokens,
        "estimated_tokens_hard_upper_char": hard_tokens,
        "token_estimate_method": "conservative_mixed_v1_cjk1_other_2chars",
        "token_estimate_notes": [
            "typical_qwen_style: CJK=1 token, other=4 chars/token",
            "conservative_mixed: CJK=1 token, other=2 chars/token (budget gate)",
            "hard_upper_char: 1 token/char; used only as a ceiling, not the gate",
            "No remote tokenizer was called.",
        ],
        "estimated_http_requests": http_requests,
        "estimated_fee_cny": round(fee, 4),
        "estimated_fee_cny_typical": round(typical_fee, 4),
        "estimated_fee_cny_hard_upper": round(hard_fee, 4),
        "price_cny_per_million_tokens": PRICE_CNY_PER_MILLION,
        "price_cny_per_1k_tokens": PRICE_CNY_PER_1K,
        "price_source": PRICE_SOURCE,
        "price_region": PRICE_REGION,
        "estimated_disk_mb": round(disk_mb, 2),
        "oversized_chunks": oversized,
        "within_token_budget": conservative_tokens <= MAX_TOKENS,
        "within_http_budget": http_requests <= MAX_HTTP,
        "within_fee_budget": fee <= MAX_FEE_CNY,
        "chunker": {
            "implementation": "lumenfin.rag.chunking.chunk_document",
            "max_chunk_chars": 900,
            "overlap_chars": 120,
        },
    }


def budget_ok(estimate: Mapping[str, Any]) -> bool:
    return bool(
        estimate.get("within_token_budget")
        and estimate.get("within_http_budget")
        and estimate.get("within_fee_budget")
        and int(estimate.get("oversized_chunks") or 0) == 0
        and int(estimate.get("chunks") or 0) > 0
    )


def credential_present() -> dict[str, Any]:
    source = "env" if os.getenv("DASHSCOPE_API_KEY") else ""
    return {
        "name": "DASHSCOPE_API_KEY",
        "source": source or "missing",
        "present": bool(source),
    }


def run_dry_run(
    *,
    repo_root: Path,
    snapshot: Path,
    official: bool = True,
    salt: str = SPLIT_SALT,
) -> dict[str, Any]:
    source_hashes = assert_rc5_sources(repo_root=repo_root)
    reports, audit, rows_seen = scan_holdout_reports(snapshot, salt=salt)
    documents, stats = materialize_documents(reports)
    identity = {
        "source_artifact_sha256": "",
        "company_keys_sha256": stats["company_keys_sha256"],
    }
    if official:
        identity = verify_official_identity(
            repo_root=repo_root, snapshot=snapshot, stats=stats
        )
    estimate = estimate_from_documents(documents)
    payload = {
        "schema_version": "ledger_public_holdout_index_dryrun.v1",
        "status": "BUDGET_OK" if budget_ok(estimate) else "BUDGET_EXCEEDED",
        "product_tag": PRODUCT_TAG,
        "product_commit": PRODUCT_COMMIT,
        "change_kind": "resource_authorization_before_holdout_consumption",
        "model_or_retrieval_tuning": False,
        "input_access_audit": audit.as_dict(),
        "corpus": {
            **stats,
            "snapshot_rows_scanned": rows_seen,
            **identity,
        },
        "estimate": estimate,
        "rc5_source_sha256": source_hashes,
        "embedding": {
            "provider": "dashscope",
            "model": "text-embedding-v4",
            "dimension": 1024,
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "batch_size": DASHSCOPE_BATCH,
        },
        "deepseek_calls_during_index_build": 0,
        "qwen3_calls_during_index_build": 0,
        "credential": credential_present(),
        "holdout_consumed": False,
    }
    return payload


def open_store(*, uri: str, allow_remote: bool):
    from ..eval.financebench.retrieval import build_eval_store

    return build_eval_store(
        uri=uri,
        embedding_provider="dashscope",
        embedding_dimension=1024,
        collection_name=COLLECTION,
        allow_remote=allow_remote,
        mode="hybrid",
        embedding_model="text-embedding-v4",
    )


def run_canary(*, store: Any, companies: list[str], expected_chunks: int) -> dict[str, Any]:
    if not companies:
        raise HoldoutIndexError("canary needs at least one company key")
    company = companies[0]
    dense = store.vector_search(
        CANARY_QUERY,
        session_id=SESSION_ID,
        tenant_id=SESSION_ID,
        companies=[company],
        top_k=5,
    )
    sparse = store.bm25_search(
        CANARY_QUERY,
        session_id=SESSION_ID,
        tenant_id=SESSION_ID,
        companies=[company],
        top_k=5,
    )
    unfiltered = store.bm25_search(
        CANARY_QUERY,
        session_id=SESSION_ID,
        tenant_id=SESSION_ID,
        top_k=5,
    )
    row_count = None
    try:
        stats = store.client.get_collection_stats(store.collection_name)
        if isinstance(stats, Mapping):
            row_count = int(stats.get("row_count") or stats.get("rowCount") or 0)
    except Exception:
        row_count = None
    if not dense:
        raise HoldoutIndexError("canary dense search returned no hits")
    if not sparse:
        raise HoldoutIndexError("canary BM25 search returned no hits")
    if not unfiltered:
        raise HoldoutIndexError("canary unfiltered BM25 search returned no hits")
    return {
        "query_kind": "synthetic_financial_keywords",
        "holdout_query_used": False,
        "company_filter": company,
        "dense_hits": len(dense),
        "bm25_hits": len(sparse),
        "unfiltered_bm25_hits": len(unfiltered),
        "row_count": row_count,
        "expected_chunks": expected_chunks,
        "collection": store.collection_name,
        "session_id": SESSION_ID,
        "tenant_id": SESSION_ID,
        "schema": "dense_bm25_v1",
    }
