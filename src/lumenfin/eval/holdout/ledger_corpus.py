"""Build a page-aligned LEDGER public-dev corpus without touching holdout scoring."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .governance import HoldoutError
from .ledger import (
    LEDGER_PAGE_DELIMITER,
    PUBLIC_DEV,
    adapt_ledger_row,
    assign_ledger_company_split,
)

_PAGE_SPLIT_RE = re.compile(re.escape(LEDGER_PAGE_DELIMITER), re.IGNORECASE)


@dataclass(frozen=True)
class LedgerPublicDevDataset:
    queries: tuple[dict[str, Any], ...]
    page_documents: tuple[dict[str, Any], ...]
    companies: tuple[str, ...]
    reports: int
    ignored_zero_qrels: int = 0
    unavailable_positive_qrels: int = 0
    queries_with_unavailable_positive_qrels: int = 0
    excluded_queries_without_reachable_positive: int = 0
    excluded_query_ids_sha256: str = ""


def ledger_public_dev_qrel_audit(
    dataset: LedgerPublicDevDataset,
    *,
    source_queries: int,
) -> dict[str, Any]:
    return {
        "policy": "drop_unindexable_blank_qrels_v1",
        "blank_pages_indexed": False,
        "source_queries": int(source_queries),
        "scorable_queries": len(dataset.queries),
        "scorable_query_ids_sha256": _ids_sha256(
            str(query["query_id"]) for query in dataset.queries
        ),
        "ignored_zero_qrels": dataset.ignored_zero_qrels,
        "unavailable_positive_qrels": dataset.unavailable_positive_qrels,
        "queries_with_unavailable_positive_qrels": (
            dataset.queries_with_unavailable_positive_qrels
        ),
        "excluded_queries_without_reachable_positive": (
            dataset.excluded_queries_without_reachable_positive
        ),
        "excluded_query_ids_sha256": dataset.excluded_query_ids_sha256,
    }


def _ids_sha256(values: Iterable[str]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _report_id(record: Mapping[str, Any]) -> str:
    return f"{record['exchange']}_{record['ticker']}_{record['year']}"


def _page_documents(
    *,
    report_id: str,
    company_key: str,
    company_name: str,
    mmd_text: str,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for page_zero, raw_page in enumerate(_PAGE_SPLIT_RE.split(mmd_text)):
        text = raw_page.strip()
        if not text:
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
    return documents


def _validate_manifest_split(
    *,
    queries: list[dict[str, Any]],
    companies: set[str],
    manifest: Mapping[str, Any],
) -> None:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise HoldoutError("LEDGER manifest needs a splits object")
    expected = splits.get(PUBLIC_DEV)
    if not isinstance(expected, Mapping):
        raise HoldoutError("LEDGER manifest needs a public_dev split")
    query_ids = [str(query["query_id"]) for query in queries]
    checks = (
        ("queries", len(query_ids)),
        ("companies", len(companies)),
        ("query_ids_sha256", _ids_sha256(query_ids)),
        ("company_keys_sha256", _ids_sha256(companies)),
    )
    for field, actual in checks:
        if expected.get(field) != actual:
            raise HoldoutError(
                f"LEDGER public_dev does not match frozen manifest field {field}"
            )
    if expected.get("local_tuning_allowed") is not True:
        raise HoldoutError("LEDGER manifest does not authorize public_dev tuning")


def build_ledger_public_dev_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    salt: str,
    holdout_fraction: float = 0.2,
    manifest: Mapping[str, Any] | None = None,
) -> LedgerPublicDevDataset:
    """Select public-dev by company, deduplicate reports, and verify every qrel page."""
    queries: list[dict[str, Any]] = []
    query_ids: set[str] = set()
    companies: set[str] = set()
    reports: dict[str, tuple[str, str, str, str]] = {}

    for raw_row in rows:
        record = adapt_ledger_row(raw_row)
        company_key = str(record["company_key"])
        if (
            assign_ledger_company_split(
                company_key,
                salt=salt,
                holdout_fraction=holdout_fraction,
            )
            != PUBLIC_DEV
        ):
            continue
        query_id = str(record["query_id"])
        if query_id in query_ids:
            raise HoldoutError("LEDGER public_dev contains duplicate query_id")
        query_ids.add(query_id)
        companies.add(company_key)
        queries.append(
            {
                "query_id": query_id,
                "query_text": str(record["query_text"]),
                "company_key": company_key,
                "qrels": tuple(dict(item) for item in record["qrels"]),
            }
        )

        report_id = _report_id(record)
        mmd_text = str(raw_row["mmd_text"])
        content_hash = hashlib.sha256(mmd_text.encode("utf-8")).hexdigest()
        previous = reports.get(report_id)
        report_identity = (content_hash, company_key, str(record["company_name"]))
        if previous is not None and previous[:3] != report_identity:
            raise HoldoutError(
                "LEDGER repeated report has inconsistent content or company identity"
            )
        reports.setdefault(report_id, (*report_identity, mmd_text))

    if not queries:
        raise HoldoutError("LEDGER public_dev selection is empty")
    if manifest is not None:
        _validate_manifest_split(
            queries=queries,
            companies=companies,
            manifest=manifest,
        )

    page_documents: list[dict[str, Any]] = []
    for report_id in sorted(reports):
        _content_hash, company_key, company_name, mmd_text = reports[report_id]
        page_documents.extend(
            _page_documents(
                report_id=report_id,
                company_key=company_key,
                company_name=company_name,
                mmd_text=mmd_text,
            )
        )
    doc_ids = {str(document["ledger_doc_id"]) for document in page_documents}
    if len(doc_ids) != len(page_documents):
        raise HoldoutError("LEDGER public_dev page corpus contains duplicate doc_id")
    missing_report_refs = 0
    blank_page_refs = 0
    out_of_range_page_refs = 0
    unexplained_page_refs = 0
    invalid_zero_qrels = 0
    ignored_zero_qrels = 0
    queries_with_missing_positive: set[str] = set()
    queries_without_reachable_positive: set[str] = set()
    scorable_queries: list[dict[str, Any]] = []
    for query in queries:
        reachable_positive = False
        available_qrels: list[dict[str, int]] = []
        for qrel in query["qrels"]:
            qrel_doc_id = str(qrel["doc_id"])
            relevance = int(qrel["relevance"])
            if qrel_doc_id in doc_ids:
                reachable_positive = reachable_positive or relevance > 0
                available_qrels.append(dict(qrel))
                continue
            qrel_report_id = qrel_doc_id.rsplit("/page_", 1)[0]
            missing_kind = ""
            if qrel_report_id not in reports:
                missing_kind = "missing_report"
            else:
                page_zero = int(qrel_doc_id.rsplit("/page_", 1)[1])
                report_pages = _PAGE_SPLIT_RE.split(reports[qrel_report_id][3])
                if page_zero >= len(report_pages):
                    missing_kind = "out_of_range"
                elif not report_pages[page_zero].strip():
                    missing_kind = "blank"
                else:
                    missing_kind = "unexplained"
            if relevance == 0:
                if missing_kind == "blank":
                    ignored_zero_qrels += 1
                else:
                    invalid_zero_qrels += 1
                continue
            queries_with_missing_positive.add(str(query["query_id"]))
            if missing_kind == "missing_report":
                missing_report_refs += 1
            elif missing_kind == "out_of_range":
                out_of_range_page_refs += 1
            elif missing_kind == "blank":
                blank_page_refs += 1
            else:
                unexplained_page_refs += 1
        if not reachable_positive:
            queries_without_reachable_positive.add(str(query["query_id"]))
            continue
        scorable_query = dict(query)
        scorable_query["qrels"] = tuple(available_qrels)
        scorable_queries.append(scorable_query)
    if (
        missing_report_refs
        or out_of_range_page_refs
        or unexplained_page_refs
        or invalid_zero_qrels
    ):
        raise HoldoutError(
            "LEDGER public_dev qrels reference invalid corpus pages "
            f"(missing_report_refs={missing_report_refs}, "
            f"out_of_range_page_refs={out_of_range_page_refs}, "
            f"unexplained_page_refs={unexplained_page_refs}, "
            f"invalid_zero_qrels={invalid_zero_qrels})"
        )
    if not scorable_queries:
        raise HoldoutError("LEDGER public_dev has no queries with reachable positive qrels")

    dataset = LedgerPublicDevDataset(
        queries=tuple(
            sorted(scorable_queries, key=lambda item: str(item["query_id"]))
        ),
        page_documents=tuple(page_documents),
        companies=tuple(sorted(companies)),
        reports=len(reports),
        ignored_zero_qrels=ignored_zero_qrels,
        unavailable_positive_qrels=blank_page_refs,
        queries_with_unavailable_positive_qrels=len(
            queries_with_missing_positive
        ),
        excluded_queries_without_reachable_positive=len(
            queries_without_reachable_positive
        ),
        excluded_query_ids_sha256=_ids_sha256(
            queries_without_reachable_positive
        ),
    )
    if manifest is not None:
        expected_audit = manifest.get("public_dev_corpus_audit")
        actual_audit = ledger_public_dev_qrel_audit(
            dataset,
            source_queries=len(queries),
        )
        if not isinstance(expected_audit, Mapping) or dict(expected_audit) != actual_audit:
            raise HoldoutError(
                "LEDGER public_dev qrel corpus audit does not match frozen manifest"
            )
    return dataset
