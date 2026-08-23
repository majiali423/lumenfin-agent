"""Independent synthetic remote alias-compliance canary.

This suite only tests whether a model can emit legal citation aliases,
whether those aliases map to stable chunk IDs, whether invalid output
fail-closes, and whether the API/FinRun contract stays atomic.

It is not product accuracy, not a benchmark, not financial accuracy, not
retrieval quality, and not a LEDGER/FinanceBench/holdout score. Official
preflight and remote execution stay default-deny until a later one-shot
authorization. Hash `51331a4f…` never preflighted or remoted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from ..api.schemas import AnalyzeResponse
from ..citation_alias import (
    CITATION_ALIAS_PROTOCOL_VERSION,
    LOCKED_FINAL_K,
    CitationAliasError,
    CitationAliasMap,
    allowlist_from_window,
    build_citation_alias_map,
    build_final_evidence_window,
    official_ranked_hits,
    parse_and_map_citation_aliases,
    render_alias_passages,
    render_prompt_evidence,
)
from ..finrun import export_finrun_state
from ..structured_answer import (
    CITATION_VALIDATION_FAILED,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    StructuredAnswerError,
    degraded_structured_answer,
    public_structured_answer_fields,
    redact_structured_error,
    validate_structured_answer,
)

SUITE = "synthetic_remote_alias_compliance_canary"
SUITE_VERSION = "1.0"
DATASET_KIND = "synthetic_fictional"
DATASET_SCHEMA_VERSION = "synthetic_alias_compliance_dataset.v1"
CONFIG_SCHEMA_VERSION = "synthetic_alias_compliance_config.v1"
AUTHORIZATION_SCHEMA_VERSION = "synthetic_alias_compliance_authorization.v1"
AUTHORIZATION_KIND = "synthetic_alias_compliance_authorization"
OUTPUT_SCHEMA_VERSION = "synthetic_alias_compliance_result.v1"
PREFLIGHT_SCHEMA_VERSION = "synthetic_alias_compliance_preflight.v1"
IDENTITY_STATUS = "SYNTHETIC_CONTRACT_IMPLEMENTATION_IDENTITY"
UNAUTHORIZED_REASON = "implementation_only_not_authorized_this_phase"
RETIRED_LEDGER_V5_HASH = (
    "7db4156491fbd0cb500ae71772002a494a3cc37b751eb5e55b707307fd02b91b"
)
RETIRED_BEFORE_PREFLIGHT_HASH = (
    "51331a4f059d02905f8c2dd61abfe9c180919140f8973303570e6095c529f793"
)
RETIRED_CONFIG_HASHES = {
    RETIRED_LEDGER_V5_HASH: "retired LEDGER V5 config hash is not executable",
    RETIRED_BEFORE_PREFLIGHT_HASH: (
        "retired synthetic alias config hash never preflighted or remoted"
    ),
}
DEFAULT_CHAT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
FICTIONAL_COMPANY = "PebblefordLanterns"
LOCKED_CASE_COUNT = 8
EXPECTED_CITATION_CASES = 7
MAX_ATTEMPTS_PER_CASE = 3
HASH_SKIP_KEYS = frozenset({"config_hash"})
AUTHORIZATION_HASH_SKIP = frozenset({"authorization_sha256"})
EVAL_GOLD_SENTINEL = "SYNTHETIC_GOLD_SENTINEL_DO_NOT_PROMPT"
DEFAULT_DATASET_PATH = Path("data") / "eval_rag" / "synthetic_alias_compliance_cases.json"
DEFAULT_CONFIG_PATH = Path("data") / "eval_rag" / "synthetic_alias_compliance_config.json"
DEFAULT_AUTHORIZATION_PATH = (
    Path("data") / "eval_rag" / "synthetic_alias_compliance_authorization.json"
)
DEFAULT_PREFLIGHT_OUTPUT_DIR = Path("outputs") / "synthetic_alias_compliance_preflight_v1"
DEFAULT_OFFICIAL_OUTPUT_DIR = Path("outputs") / "synthetic_alias_compliance_v1"
FORBIDDEN_PATH_TOKENS = (
    "ledger",
    "financebench",
    "holdout",
    "public_dev",
    "public_holdout",
    "structured_citation_shadow",
    "ledger_structured_citation",
)
FORBIDDEN_CLI_FLAGS = {
    "--dataset",
    "--dataset-path",
    "--cases-path",
    "--cases_path",
    "--split",
    "--limit",
    "--model",
    "--prompt",
    "--final-k",
    "--final_k",
    "--top-k",
    "--top_k",
    "--output-dir",
    "--preflight-dir",
    "--frozen-config",
    "--rag",
    "--embedding",
    "--reranker",
    "--snapshot",
    "--parquet-path",
}
FORCE_ENV_KEYS = (
    "SYNTHETIC_ALIAS_COMPLIANCE_FORCE",
    "LUMENFIN_FORCE_REMOTE",
    "LUMENFIN_SYNTHETIC_ALIAS_FORCE",
)
LEDGER_CASE_ID_RE = re.compile(r"^[A-Z]{1,5}_[a-z0-9_]+_\d{4}$")
REAL_WORLD_NEEDLES = (
    "apple",
    "microsoft",
    "alphabet",
    "google",
    "amazon",
    "nvidia",
    "tesla",
    "meta platforms",
    "berkshire",
    "jpmorgan",
    "goldman",
    "financebench",
    "public_holdout",
    "public_dev",
    "10-k",
    "10-q",
    "sec.gov",
)
SYNTHETIC_ALIAS_SYSTEM_PROMPT = (
    "Read the numbered passages. "
    "Reply with JSON only: "
    '{"answer": <string>, "citations": ["E01"], '
    '"structured_answer_schema_version": "1.0", '
    '"abstain": <bool>}. '
    "Citations must be exact evidence aliases such as E01 from the passages. "
    "Do not emit chunk ids, filenames, page numbers, or raw digits. "
    "If the passages do not contain the answer, set abstain=true and "
    "citations to []. "
    "Do not invent numbers or aliases."
)


class CanaryError(ValueError):
    """Fail-closed synthetic alias-compliance error. Safe to log."""


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_normalized_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def compute_config_hash(payload: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_dumps({key: value for key, value in payload.items() if key not in HASH_SKIP_KEYS})
    )


def compute_authorization_hash(payload: Mapping[str, Any]) -> str:
    return sha256_text(
        canonical_dumps(
            {key: value for key, value in payload.items() if key not in AUTHORIZATION_HASH_SKIP}
        )
    )


def prompt_sha256(text: str | None = None) -> str:
    return sha256_text(text or SYNTHETIC_ALIAS_SYSTEM_PROMPT)


def chat_base_url_sha256(url: str = DEFAULT_CHAT_BASE_URL) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def _strict_true(value: Any) -> bool:
    return value is True


def _posix(path: Path | str) -> str:
    return Path(path).as_posix()


def repo_rel(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    root = repo_root.resolve()
    return resolved.relative_to(root).as_posix()


def assert_isolated_path(path: Path | str, *, field: str) -> Path:
    target = Path(path)
    lowered = _posix(target).casefold()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise CanaryError(f"{field} refuses path token {token}")
    return target


def assert_no_network_dataset_source(path: Path) -> None:
    text = str(path)
    if text.startswith(("http://", "https://", "hf://")):
        raise CanaryError("dataset download is forbidden")


def assert_case_id_isolated(case_id: str) -> None:
    value = str(case_id or "").strip()
    if not value.startswith("SAC-"):
        raise CanaryError("synthetic case id must use the SAC- prefix")
    if LEDGER_CASE_ID_RE.match(value):
        raise CanaryError("case id overlaps LEDGER format")
    lowered = value.casefold()
    for token in ("ledger", "financebench", "holdout", "public_dev", "fb_"):
        if token in lowered:
            raise CanaryError("case id overlaps a reserved eval identity")


def assert_synthetic_text(blob: str, *, field: str) -> None:
    lowered = str(blob or "").casefold()
    for needle in REAL_WORLD_NEEDLES:
        if needle in lowered:
            raise CanaryError(f"{field} contains reserved real-world token {needle}")


def git_snapshot(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        result = subprocess.run(
            args,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or "").strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    return {
        "lumenfin_commit": commit or "unknown",
        "worktree_dirty": bool(porcelain),
        "worktree_status": "dirty" if porcelain else "clean",
    }


@dataclass(frozen=True)
class SyntheticCase:
    case_id: str
    query_text: str
    passages: tuple[dict[str, str], ...]
    expected_aliases: tuple[str, ...]
    gold_answer: str | None
    incomplete: bool

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(item["chunk_id"] for item in self.passages)

    @property
    def qrels(self) -> dict[str, int]:
        if self.incomplete:
            return {}
        alias_to_chunk = {
            f"E{index:02d}": passage["chunk_id"]
            for index, passage in enumerate(self.passages, start=1)
        }
        return {alias_to_chunk[alias]: 1 for alias in self.expected_aliases}


@dataclass(frozen=True)
class BoundCase:
    case: SyntheticCase
    window: list[dict[str, Any]]
    alias_map: CitationAliasMap
    prompt_evidence: str
    generation_view: dict[str, Any]
    evaluation_view: dict[str, Any]
    provider_request: dict[str, Any]


@dataclass(frozen=True)
class FrozenCanaryConfig:
    path: Path
    payload: dict[str, Any]
    config_hash: str

    def field(self, *keys: str, default: Any = None) -> Any:
        cursor: Any = self.payload
        for key in keys:
            if not isinstance(cursor, Mapping) or key not in cursor:
                return default
            cursor = cursor[key]
        return cursor


def read_json_object(path: Path, *, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CanaryError(f"{field} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise CanaryError(f"{field} must be a JSON object")
    return payload


def load_dataset_payload(
    path: Path | None = None,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root or Path.cwd())
    dataset_path = Path(path or (root / DEFAULT_DATASET_PATH))
    assert_no_network_dataset_source(dataset_path)
    assert_isolated_path(dataset_path, field="dataset")
    payload = read_json_object(dataset_path, field="dataset")
    if payload.get("dataset_schema_version") != DATASET_SCHEMA_VERSION:
        raise CanaryError("dataset schema_version is unsupported")
    if payload.get("suite") != SUITE:
        raise CanaryError("dataset suite mismatch")
    if payload.get("dataset_kind") != DATASET_KIND:
        raise CanaryError("dataset_kind must be synthetic_fictional")
    for flag in (
        "product_accuracy_claim",
        "benchmark_claim",
        "financial_accuracy_claim",
        "retrieval_quality_claim",
    ):
        if payload.get(flag) is not False:
            raise CanaryError(f"dataset must set {flag} false")
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != LOCKED_CASE_COUNT:
        raise CanaryError("dataset must contain exactly 8 cases")
    return payload


def parse_cases(payload: Mapping[str, Any]) -> list[SyntheticCase]:
    cases: list[SyntheticCase] = []
    for raw in payload.get("cases") or []:
        if not isinstance(raw, Mapping):
            raise CanaryError("dataset case must be an object")
        case_id = str(raw.get("case_id") or "").strip()
        assert_case_id_isolated(case_id)
        passages_raw = raw.get("passages")
        if not isinstance(passages_raw, list) or len(passages_raw) != LOCKED_FINAL_K:
            raise CanaryError(f"{case_id} must contain exactly 10 passages")
        passages: list[dict[str, str]] = []
        for item in passages_raw:
            if not isinstance(item, Mapping):
                raise CanaryError(f"{case_id} passage must be an object")
            chunk_id = str(item.get("chunk_id") or "").strip()
            text = str(item.get("text") or "")
            if not chunk_id or not text:
                raise CanaryError(f"{case_id} passage is missing chunk_id or text")
            if chunk_id in text:
                raise CanaryError(f"{case_id} passage text leaked its chunk_id")
            passages.append({"chunk_id": chunk_id, "text": text})
        expected = raw.get("expected_aliases") or []
        if not isinstance(expected, list) or not all(isinstance(item, str) for item in expected):
            raise CanaryError(f"{case_id} expected_aliases must be a string array")
        incomplete = bool(raw.get("incomplete"))
        gold = raw.get("gold_answer")
        if gold is not None and not isinstance(gold, str):
            raise CanaryError(f"{case_id} gold_answer must be a string or null")
        case = SyntheticCase(
            case_id=case_id,
            query_text=str(raw.get("query_text") or ""),
            passages=tuple(passages),
            expected_aliases=tuple(str(item) for item in expected),
            gold_answer=None if gold is None else str(gold),
            incomplete=incomplete,
        )
        assert_synthetic_text(
            " ".join(
                [
                    case.case_id,
                    case.query_text,
                    case.gold_answer or "",
                    " ".join(item["text"] for item in case.passages),
                ]
            ),
            field=case.case_id,
        )
        if case.incomplete and case.expected_aliases:
            raise CanaryError(f"{case_id} incomplete case cannot expect citations")
        if not case.incomplete and not case.expected_aliases:
            raise CanaryError(f"{case_id} citation case is missing expected aliases")
        cases.append(case)
    return cases


def dataset_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    cases = parse_cases(payload)
    case_ids = [case.case_id for case in cases]
    queries = [case.query_text for case in cases]
    passages = [
        {
            "case_id": case.case_id,
            "passages": [{"chunk_id": item["chunk_id"], "text": item["text"]} for item in case.passages],
        }
        for case in cases
    ]
    gold = [
        {
            "case_id": case.case_id,
            "expected_aliases": list(case.expected_aliases),
            "gold_answer": case.gold_answer,
            "incomplete": case.incomplete,
            "qrels": case.qrels,
        }
        for case in cases
    ]
    cases_payload = [
        {
            "case_id": case.case_id,
            "query_text": case.query_text,
            "expected_aliases": list(case.expected_aliases),
            "gold_answer": case.gold_answer,
            "incomplete": case.incomplete,
            "passages": [dict(item) for item in case.passages],
        }
        for case in cases
    ]
    identity = {
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "case_count": len(cases),
        "case_ids_sha256": sha256_text("\n".join(case_ids)),
        "query_texts_sha256": sha256_text("\n".join(queries)),
        "passages_sha256": sha256_text(canonical_dumps(passages)),
        "gold_identity_sha256": sha256_text(canonical_dumps(gold)),
        "dataset_sha256": sha256_text(canonical_dumps(cases_payload)),
    }
    if identity["case_count"] != LOCKED_CASE_COUNT:
        raise CanaryError("dataset identity case_count must stay 8")
    return identity


def load_cases(*, repo_root: Path | None = None, path: Path | None = None) -> list[SyntheticCase]:
    return parse_cases(load_dataset_payload(path, repo_root=repo_root))


def bind_case(case: SyntheticCase) -> BoundCase:
    hits = official_ranked_hits(
        [
            {
                "chunk_id": passage["chunk_id"],
                "text": passage["text"],
                "company": FICTIONAL_COMPANY,
            }
            for passage in case.passages
        ],
        origin="test_fixture",
        company_order=(FICTIONAL_COMPANY,),
    )
    window = build_final_evidence_window(hits, final_k=LOCKED_FINAL_K)
    alias_map = build_citation_alias_map(window, case_id=case.case_id)
    prompt_evidence = render_prompt_evidence(window, alias_map, query_text=case.query_text)
    render_alias_passages(
        tuple((alias, str(hit.get("text") or "")) for alias, hit in zip(alias_map.aliases, window)),
        query_text=case.query_text,
    )
    generation_view = {
        "case_id": case.case_id,
        "query_text": case.query_text,
        "passages": [
            {"alias": alias, "text": str(hit.get("text") or "")}
            for alias, hit in zip(alias_map.aliases, window)
        ],
        "structured_output_instructions": SYNTHETIC_ALIAS_SYSTEM_PROMPT,
    }
    evaluation_view = {
        "case_id": case.case_id,
        "expected_aliases": list(case.expected_aliases),
        "gold_answer": case.gold_answer,
        "incomplete": case.incomplete,
        "qrels": case.qrels,
        "chunk_ids": list(case.chunk_ids),
        "support_label": "synthetic_qrel",
        "evaluator_sentinel": EVAL_GOLD_SENTINEL,
        "gold_sentinel": f"{EVAL_GOLD_SENTINEL}:{case.case_id}:{case.gold_answer}",
    }
    provider_request = {
        "case_id": generation_view["case_id"],
        "query_text": generation_view["query_text"],
        "passages": generation_view["passages"],
        "structured_output_instructions": generation_view["structured_output_instructions"],
        "user_prompt": prompt_evidence,
    }
    _assert_generation_isolation(provider_request, evaluation_view, window)
    return BoundCase(
        case=case,
        window=window,
        alias_map=alias_map,
        prompt_evidence=prompt_evidence,
        generation_view=generation_view,
        evaluation_view=evaluation_view,
        provider_request=provider_request,
    )


def _blob(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    return canonical_dumps(payload)


def _assert_generation_isolation(
    provider_request: Mapping[str, Any],
    evaluation_view: Mapping[str, Any],
    window: list[dict[str, Any]],
) -> None:
    blob = _blob(provider_request)
    lowered = blob.casefold()
    for hit in window:
        chunk_id = str(hit.get("chunk_id") or "")
        if chunk_id and chunk_id in blob:
            raise CanaryError("provider request leaked a stable chunk id")
    for key in (
        "gold_answer",
        "expected_aliases",
        "qrels",
        "support_label",
        "evaluator_sentinel",
        "gold_sentinel",
        "chunk_ids",
    ):
        if key in provider_request:
            raise CanaryError("provider request included evaluator metadata")
    for token in (
        EVAL_GOLD_SENTINEL,
        str(evaluation_view.get("gold_sentinel") or ""),
        "qrels",
        "gold_answer",
        "expected_aliases",
        "support_label",
    ):
        if token and token.casefold() in lowered:
            raise CanaryError("provider request leaked gold or evaluator metadata")
    allowed_keys = {
        "case_id",
        "query_text",
        "passages",
        "structured_output_instructions",
        "user_prompt",
    }
    extra = set(provider_request) - allowed_keys
    if extra:
        raise CanaryError("provider request contained unsupported fields")


def classify_raw_citations(raw: object, alias_map: CitationAliasMap) -> dict[str, Any]:
    report = {
        "json_parse_success": False,
        "alias_syntax_valid": False,
        "alias_allowlist_valid": False,
        "raw_chunk_id_leakage": False,
        "unknown_aliases": False,
        "mixed_valid_invalid": False,
        "alias_mapping_success": False,
        "mapped_chunk_ids": [],
        "error": "",
    }
    if not isinstance(raw, list):
        report["error"] = "citations must be a string array"
        return report
    tokens = [str(item) for item in raw]
    chunk_ids = set(alias_map.chunk_ids)
    valid: list[str] = []
    invalid = False
    unknown = False
    syntax_ok = True
    for item in raw:
        if isinstance(item, str) and item.strip() in chunk_ids:
            report["raw_chunk_id_leakage"] = True
            invalid = True
            continue
        try:
            from ..citation_alias import normalize_alias_token

            token = normalize_alias_token(item)
        except CitationAliasError:
            syntax_ok = False
            invalid = True
            continue
        if token not in alias_map.alias_to_chunk:
            unknown = True
            invalid = True
            continue
        valid.append(token)
    report["alias_syntax_valid"] = syntax_ok and not report["raw_chunk_id_leakage"]
    report["unknown_aliases"] = unknown
    report["mixed_valid_invalid"] = bool(valid) and invalid
    report["alias_allowlist_valid"] = bool(syntax_ok and not invalid)
    if report["mixed_valid_invalid"] or report["raw_chunk_id_leakage"] or unknown or not syntax_ok:
        return report
    try:
        mapped = parse_and_map_citation_aliases(tokens, alias_map, expected_case_id=alias_map.case_id)
    except CitationAliasError as exc:
        report["error"] = redact_structured_error(str(exc))
        return report
    report["alias_mapping_success"] = True
    report["mapped_chunk_ids"] = mapped
    return report


def parse_model_json(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CanaryError("model output is not JSON") from exc
    if not isinstance(payload, dict):
        raise CanaryError("model output must be a JSON object")
    return payload


def _contract_state(
    bound: BoundCase,
    structured: Mapping[str, Any],
    *,
    workflow_status: str,
) -> dict[str, Any]:
    company = FICTIONAL_COMPANY
    hits = [dict(item) for item in bound.window]
    claims = []
    for index, chunk_id in enumerate(structured.get("citations") or [], start=1):
        claims.append(
            {
                "claim_id": f"{bound.case.case_id}-c{index}",
                "entity": company,
                "claim_type": "risk_conclusion",
                "statement": str(structured.get("answer") or bound.case.query_text),
                "verification": "verified",
                "evidence_refs": [
                    {
                        "evidence_id": f"{bound.case.case_id}-e{index}",
                        "entity": company,
                        "citation": "",
                        "source_type": "document_extracted",
                        "text": "",
                        "chunk_id": chunk_id,
                    }
                ],
            }
        )
    return {
        "run_id": bound.case.case_id,
        "thread_id": bound.case.case_id,
        "query": bound.case.query_text,
        "final_report": str(structured.get("answer") or ""),
        "workflow_status": workflow_status,
        "llm_backend": "deepseek",
        "companies": [company],
        "rag_evidence": {company: hits},
        "citation_final_window": {"hits": hits},
        "claims": claims,
        "verified_claims": claims,
        "structured_answer": dict(structured),
        "tenant_id": "synthetic-alias-compliance",
        "rag_tenant_id": "synthetic-alias-compliance",
    }


def score_bound_output(bound: BoundCase, raw_text: str) -> dict[str, Any]:
    metrics = {
        "json_parse_success": False,
        "alias_syntax_valid": False,
        "alias_allowlist_valid": False,
        "raw_chunk_id_leakage": False,
        "unknown_aliases": False,
        "mixed_valid_invalid": False,
        "alias_mapping_success": False,
        "structured_answer_emitted": False,
        "citation_validation_failed": False,
        "expected_alias_match": False,
        "mapped_chunk_matches_synthetic_qrels": False,
        "incomplete_case_handled": False,
        "public_triple_emitted": False,
        "finrun_validation": "",
    }
    try:
        payload = parse_model_json(raw_text)
        metrics["json_parse_success"] = True
    except CanaryError:
        return _failed_contract(
            bound,
            metrics,
            answer="",
            workflow_status="incomplete_data" if bound.case.incomplete else "completed",
            error="model output is not JSON",
            finrun_citations=["protocol_failure"],
        )

    citations = payload.get("citations")
    alias_report = classify_raw_citations(citations, bound.alias_map)
    for key in (
        "alias_syntax_valid",
        "alias_allowlist_valid",
        "raw_chunk_id_leakage",
        "unknown_aliases",
        "mixed_valid_invalid",
        "alias_mapping_success",
    ):
        metrics[key] = alias_report[key]

    abstain = bool(payload.get("abstain"))
    workflow_status = "incomplete_data" if bound.case.incomplete or abstain else "completed"
    allowed = allowlist_from_window(bound.window, verified_ids=bound.case.chunk_ids)
    if alias_report["alias_mapping_success"]:
        try:
            structured_obj = validate_structured_answer(
                {
                    "answer": payload.get("answer", ""),
                    "citations": alias_report["mapped_chunk_ids"],
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                    "workflow_status": workflow_status,
                },
                allowed=allowed,
                require_citation_for_factual=not bound.case.incomplete,
            )
            if bound.case.incomplete and not alias_report["mapped_chunk_ids"]:
                metrics["alias_mapping_success"] = False
                metrics["incomplete_case_handled"] = True
                metrics["alias_syntax_valid"] = True
                metrics["alias_allowlist_valid"] = True
            return _finished_contract(
                bound,
                metrics,
                structured_obj.to_dict(),
                workflow_status=workflow_status,
            )
        except StructuredAnswerError as exc:
            return _failed_contract(
                bound,
                metrics,
                answer=str(payload.get("answer") or ""),
                workflow_status=workflow_status,
                error=exc,
                finrun_citations=["protocol_failure"],
            )

    if (
        bound.case.incomplete
        and not alias_report["raw_chunk_id_leakage"]
        and not alias_report["unknown_aliases"]
        and not alias_report["mixed_valid_invalid"]
        and list(payload.get("citations") or []) == []
        and (abstain or workflow_status == "incomplete_data")
    ):
        try:
            structured_obj = validate_structured_answer(
                {
                    "answer": payload.get("answer", ""),
                    "citations": [],
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                    "workflow_status": "incomplete_data",
                },
                allowed=allowed,
                require_citation_for_factual=False,
            )
            metrics["incomplete_case_handled"] = True
            metrics["alias_syntax_valid"] = True
            metrics["alias_allowlist_valid"] = True
            return _finished_contract(
                bound,
                metrics,
                structured_obj.to_dict(),
                workflow_status="incomplete_data",
            )
        except StructuredAnswerError as exc:
            return _failed_contract(
                bound,
                metrics,
                answer=str(payload.get("answer") or ""),
                workflow_status="incomplete_data",
                error=exc,
                finrun_citations=["protocol_failure"],
            )

    illegal = []
    if isinstance(citations, list):
        illegal = [
            str(item)
            for item in citations
            if not (isinstance(item, str) and item.strip() in set(bound.case.chunk_ids))
        ]
    return _failed_contract(
        bound,
        metrics,
        answer=str(payload.get("answer") or ""),
        workflow_status=workflow_status,
        error=alias_report.get("error") or "citation aliases failed closed",
        finrun_citations=illegal or ["protocol_failure"],
    )


def _finished_contract(
    bound: BoundCase,
    metrics: dict[str, Any],
    structured: Mapping[str, Any],
    *,
    workflow_status: str,
) -> dict[str, Any]:
    metrics["structured_answer_emitted"] = True
    expected_chunks = [bound.alias_map.alias_to_chunk[alias] for alias in bound.case.expected_aliases]
    mapped = list(structured.get("citations") or [])
    metrics["expected_alias_match"] = (not bound.case.incomplete) and mapped == expected_chunks
    metrics["mapped_chunk_matches_synthetic_qrels"] = (
        not bound.case.incomplete
        and bool(mapped)
        and all(bound.case.qrels.get(chunk_id) == 1 for chunk_id in mapped)
        and set(mapped) == set(bound.case.qrels)
    )
    return _emit_contract(bound, metrics, structured, workflow_status=workflow_status)


def _failed_contract(
    bound: BoundCase,
    metrics: dict[str, Any],
    *,
    answer: str,
    workflow_status: str,
    error: Any,
    finrun_citations: list[str],
) -> dict[str, Any]:
    metrics["citation_validation_failed"] = True
    public_structured = degraded_structured_answer(
        answer=answer,
        workflow_status=workflow_status,
        error=error,
    )
    finrun_structured = {
        "answer": answer,
        "citations": list(finrun_citations),
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "workflow_status": workflow_status,
    }
    result = _emit_contract(
        bound,
        metrics,
        public_structured,
        workflow_status=workflow_status,
        finrun_structured=finrun_structured,
    )
    if result["metrics"]["finrun_validation"] != CITATION_VALIDATION_FAILED:
        raise CanaryError("FinRun must record citation_validation=failed")
    if result["public_fields"] is not None:
        raise CanaryError("failed structured answer must omit the API triple")
    return result


def _emit_contract(
    bound: BoundCase,
    metrics: dict[str, Any],
    structured: Mapping[str, Any],
    *,
    workflow_status: str,
    finrun_structured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    public = public_structured_answer_fields(
        {"structured_answer": structured, "final_report": str(structured.get("answer") or "")}
    )
    if public is None:
        metrics["public_triple_emitted"] = False
    else:
        metrics["public_triple_emitted"] = True
        if any(alias in set(public["citations"]) for alias in bound.alias_map.aliases):
            raise CanaryError("public citations leaked ephemeral aliases")
        if any(chunk_id not in set(bound.case.chunk_ids) for chunk_id in public["citations"]):
            raise CanaryError("public citations must be stable chunk IDs")
    state = _contract_state(
        bound,
        finrun_structured or structured,
        workflow_status=str((finrun_structured or structured).get("workflow_status") or workflow_status),
    )
    finrun = export_finrun_state(state)
    metrics["finrun_validation"] = str((finrun.get("metadata") or {}).get("citation_validation") or "")
    return {
        "metrics": metrics,
        "structured_answer": dict(structured),
        "public_fields": public,
        "api": _api_payload(state, public),
        "finrun": finrun,
    }


def _api_payload(state: Mapping[str, Any], public: Mapping[str, Any] | None) -> dict[str, Any]:
    response = AnalyzeResponse(
        thread_id=str(state.get("thread_id") or ""),
        llm_backend=str(state.get("llm_backend") or "deepseek"),
        workflow_status=str(state.get("workflow_status") or "completed"),
        final_report=str(state.get("final_report") or ""),
        audit_log=[],
        artifacts={},
        state={},
        answer=None if public is None else public["answer"],
        citations=[] if public is None else list(public["citations"]),
        structured_answer_schema_version=(
            None if public is None else public["structured_answer_schema_version"]
        ),
    )
    dumped = response.model_dump()
    if public is None:
        if dumped.get("answer") is not None or dumped.get("structured_answer_schema_version") is not None:
            raise CanaryError("failed structured answer leaked a partial API triple")
    else:
        if (
            dumped.get("answer") is None
            or dumped.get("structured_answer_schema_version") is None
            or not isinstance(dumped.get("citations"), list)
        ):
            raise CanaryError("valid structured answer must emit an atomic API triple")
    return dumped


def empty_metrics() -> dict[str, int | bool]:
    return {
        "cases_total": 0,
        "cases_succeeded": 0,
        "cases_failed": 0,
        "logical_generate_calls": 0,
        "recorded_remote_calls": 0,
        "generate_attempts": 0,
        "provider_errors": 0,
        "latency_p50": 0.0,
        "latency_p95": 0.0,
        "json_parse_success": 0,
        "alias_syntax_valid": 0,
        "alias_allowlist_valid": 0,
        "raw_chunk_id_leakage": 0,
        "unknown_aliases": 0,
        "mixed_valid_invalid": 0,
        "alias_mapping_success": 0,
        "structured_answer_emitted": 0,
        "citation_validation_failed": 0,
        "expected_alias_match": 0,
        "mapped_chunk_matches_synthetic_qrels": 0,
        "incomplete_case_handled": False,
        "protocol_gate_passed": False,
        "synthetic_evidence_gate_passed": False,
        "release_gate_passed": False,
    }


def apply_case_metrics(total: dict[str, Any], case_metrics: Mapping[str, Any]) -> None:
    for key in (
        "json_parse_success",
        "alias_syntax_valid",
        "alias_allowlist_valid",
        "raw_chunk_id_leakage",
        "unknown_aliases",
        "mixed_valid_invalid",
        "alias_mapping_success",
        "structured_answer_emitted",
        "citation_validation_failed",
        "expected_alias_match",
        "mapped_chunk_matches_synthetic_qrels",
    ):
        total[key] = int(total.get(key) or 0) + (1 if case_metrics.get(key) else 0)
    if case_metrics.get("incomplete_case_handled"):
        total["incomplete_case_handled"] = True


def evaluate_protocol_gate(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("provider_errors") or 0) == 0
        and int(metrics.get("json_parse_success") or 0) == LOCKED_CASE_COUNT
        and int(metrics.get("unknown_aliases") or 0) == 0
        and int(metrics.get("raw_chunk_id_leakage") or 0) == 0
        and int(metrics.get("mixed_valid_invalid") or 0) == 0
        and int(metrics.get("alias_mapping_success") or 0) == EXPECTED_CITATION_CASES
        and int(metrics.get("citation_validation_failed") or 0) == 0
        and metrics.get("incomplete_case_handled") is True
    )


def evaluate_synthetic_evidence_gate(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("expected_alias_match") or 0) == EXPECTED_CITATION_CASES
        and int(metrics.get("mapped_chunk_matches_synthetic_qrels") or 0) == EXPECTED_CITATION_CASES
        and metrics.get("incomplete_case_handled") is True
    )


def evaluate_release_gate(metrics: Mapping[str, Any]) -> bool:
    return (
        evaluate_protocol_gate(metrics)
        and evaluate_synthetic_evidence_gate(metrics)
        and int(metrics.get("provider_errors") or 0) == 0
        and int(metrics.get("cases_succeeded") or 0) == LOCKED_CASE_COUNT
        and int(metrics.get("cases_failed") or 0) == 0
    )


def frozen_protocol_gates() -> dict[str, Any]:
    return {
        "provider_errors": 0,
        "json_parse_success": LOCKED_CASE_COUNT,
        "unknown_aliases": 0,
        "raw_chunk_id_leakage": 0,
        "mixed_valid_invalid": 0,
        "alias_mapping_success": EXPECTED_CITATION_CASES,
        "citation_validation_failed": 0,
        "incomplete_case_handled": True,
    }


def frozen_synthetic_evidence_gates() -> dict[str, Any]:
    return {
        "expected_alias_match": EXPECTED_CITATION_CASES,
        "mapped_chunk_matches_synthetic_qrels": EXPECTED_CITATION_CASES,
        "incomplete_case_handled": True,
        "incomplete_citations_empty": True,
    }


def frozen_gate_thresholds() -> dict[str, Any]:
    return {
        "protocol": frozen_protocol_gates(),
        "synthetic_evidence": frozen_synthetic_evidence_gates(),
        "prompt_retuning_after_result": False,
        "model_selection_allowed": False,
    }


def canonical_frozen_payload(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "dataset_kind": DATASET_KIND,
        "protocol": CITATION_ALIAS_PROTOCOL_VERSION,
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "financial_accuracy_claim": False,
        "retrieval_quality_claim": False,
        "model_selection_allowed": False,
        "prompt_retuning_after_result": False,
        "dataset": {
            "path": DEFAULT_DATASET_PATH.as_posix(),
            **dict(identity),
        },
        "alias_protocol": {
            "version": CITATION_ALIAS_PROTOCOL_VERSION,
            "final_k": LOCKED_FINAL_K,
            "raw_chunk_id_output_forbidden": True,
        },
        "prompt": {
            "sha256": prompt_sha256(),
        },
        "provider": {
            "name": "deepseek",
            "model": DEFAULT_MODEL,
            "chat_base_url_sha256": chat_base_url_sha256(),
            "timeout_seconds": 60,
            "max_retries": 2,
            "concurrency": 1,
        },
        "call_budget": {
            "cases": LOCKED_CASE_COUNT,
            "logical_generate_calls_expected": LOCKED_CASE_COUNT,
            "max_attempts_per_case": MAX_ATTEMPTS_PER_CASE,
            "embedding": 0,
            "rerank": 0,
            "at_least_once": True,
            "exactly_once": False,
            "unobserved_inflight_possible": True,
        },
        "output": {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "preflight_dir": DEFAULT_PREFLIGHT_OUTPUT_DIR.as_posix(),
            "official_dir": DEFAULT_OFFICIAL_OUTPUT_DIR.as_posix(),
        },
        "execution": {
            "require_clean_commit": True,
            "require_authorization_record": True,
        },
        "gates": frozen_gate_thresholds(),
    }


def _validate_frozen_payload(payload: Mapping[str, Any], *, identity: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise CanaryError("frozen config schema_version is unsupported")
    if payload.get("suite") != SUITE:
        raise CanaryError("frozen config suite mismatch")
    if payload.get("dataset_kind") != DATASET_KIND:
        raise CanaryError("frozen config dataset_kind mismatch")
    if payload.get("protocol") != CITATION_ALIAS_PROTOCOL_VERSION:
        raise CanaryError("frozen config protocol mismatch")
    if payload.get("structured_answer_schema_version") != STRUCTURED_ANSWER_SCHEMA_VERSION:
        raise CanaryError("frozen config structured_answer_schema_version mismatch")
    for flag in (
        "product_accuracy_claim",
        "benchmark_claim",
        "financial_accuracy_claim",
        "retrieval_quality_claim",
        "model_selection_allowed",
        "prompt_retuning_after_result",
    ):
        if payload.get(flag) is not False:
            raise CanaryError(f"frozen config must set {flag} false")
    dataset = payload.get("dataset")
    if not isinstance(dataset, Mapping):
        raise CanaryError("frozen config dataset block is missing")
    if dataset.get("path") != DEFAULT_DATASET_PATH.as_posix():
        raise CanaryError("frozen config dataset path is not the published synthetic file")
    for key, value in identity.items():
        if dataset.get(key) != value:
            raise CanaryError(f"frozen config dataset {key} mismatch")
    if payload.get("prompt", {}).get("sha256") != prompt_sha256():
        raise CanaryError("frozen config prompt hash mismatch")
    provider = payload.get("provider") or {}
    if provider.get("name") != "deepseek" or provider.get("model") != DEFAULT_MODEL:
        raise CanaryError("frozen config provider/model mismatch")
    if provider.get("chat_base_url_sha256") != chat_base_url_sha256():
        raise CanaryError("frozen config endpoint hash mismatch")
    if int(provider.get("concurrency") or 0) != 1:
        raise CanaryError("frozen config concurrency must stay 1")
    budget = payload.get("call_budget") or {}
    if int(budget.get("cases") or 0) != LOCKED_CASE_COUNT:
        raise CanaryError("frozen config case budget mismatch")
    if int(budget.get("logical_generate_calls_expected") or 0) != LOCKED_CASE_COUNT:
        raise CanaryError("frozen config logical call budget mismatch")
    if int(budget.get("max_attempts_per_case") or 0) != MAX_ATTEMPTS_PER_CASE:
        raise CanaryError("frozen config attempt budget mismatch")
    if int(budget.get("embedding", -1)) != 0 or int(budget.get("rerank", -1)) != 0:
        raise CanaryError("frozen config forbids embedding and rerank")
    output = payload.get("output") or {}
    if output.get("preflight_dir") != DEFAULT_PREFLIGHT_OUTPUT_DIR.as_posix():
        raise CanaryError("frozen config preflight directory mismatch")
    if output.get("official_dir") != DEFAULT_OFFICIAL_OUTPUT_DIR.as_posix():
        raise CanaryError("frozen config official directory mismatch")
    if payload.get("gates") != frozen_gate_thresholds():
        raise CanaryError("frozen protocol gates must not be retuned")
    blob = canonical_dumps(payload)
    for token in ("financebench", "public_holdout", "public_dev", "ledger_structured", "holdout/"):
        if token in blob.casefold():
            raise CanaryError("frozen config referenced a reserved eval artifact")


def load_frozen_config(
    path: str | Path,
    *,
    repo_root: Path | None = None,
    require_published: bool = False,
) -> FrozenCanaryConfig:
    root = Path(repo_root or Path.cwd())
    config_path = Path(path)
    assert_isolated_path(config_path, field="frozen-config")
    payload = read_json_object(config_path, field="frozen-config")
    stored = str(payload.get("config_hash") or "").strip()
    computed = compute_config_hash(payload)
    if not stored:
        raise CanaryError("frozen config is missing config_hash")
    if stored != computed:
        raise CanaryError("frozen config_hash does not match canonical digest")
    if stored in RETIRED_CONFIG_HASHES:
        raise CanaryError(RETIRED_CONFIG_HASHES[stored])
    identity = dataset_identity(load_dataset_payload(repo_root=root))
    _validate_frozen_payload(payload, identity=identity)
    if require_published:
        published = root / DEFAULT_CONFIG_PATH
        if config_path.resolve() != published.resolve():
            raise CanaryError("copied or relocated frozen config is not published")
        if stored != published_config_hash(repo_root=root):
            raise CanaryError("frozen config_hash is not the published digest")
    return FrozenCanaryConfig(config_path, payload, stored)


def published_config_hash(*, repo_root: Path | None = None) -> str:
    root = Path(repo_root or Path.cwd())
    payload = read_json_object(root / DEFAULT_CONFIG_PATH, field="published-config")
    return compute_config_hash(payload)


def load_authorization(*, repo_root: Path | None = None, path: Path | None = None) -> dict[str, Any] | None:
    root = Path(repo_root or Path.cwd())
    auth_path = Path(path or (root / DEFAULT_AUTHORIZATION_PATH))
    try:
        assert_isolated_path(auth_path, field="authorization")
        payload = read_json_object(auth_path, field="authorization")
    except CanaryError:
        return None
    if payload.get("schema_version") != AUTHORIZATION_SCHEMA_VERSION:
        return None
    if payload.get("kind") != AUTHORIZATION_KIND:
        return None
    stored = str(payload.get("authorization_sha256") or "")
    computed = compute_authorization_hash(payload)
    if stored != computed:
        return None
    records_raw = payload.get("records")
    if not isinstance(records_raw, dict) or not records_raw:
        return None
    records: dict[str, dict[str, Any]] = {}
    for key, raw in records_raw.items():
        if not isinstance(raw, Mapping):
            return None
        if str(raw.get("config_hash") or "") != str(key):
            return None
        for flag in ("execution_authorized", "preflight_authorized", "remote_run_authorized"):
            if not isinstance(raw.get(flag), bool):
                return None
        records[str(key)] = dict(raw)
    return {
        "schema_version": AUTHORIZATION_SCHEMA_VERSION,
        "kind": AUTHORIZATION_KIND,
        "authorization_sha256": computed,
        "records": records,
    }


def authorization_record(
    config_hash: str,
    *,
    repo_root: Path | None = None,
    path: Path | None = None,
) -> dict[str, Any] | None:
    payload = load_authorization(repo_root=repo_root, path=path)
    if payload is None:
        return None
    record = payload["records"].get(str(config_hash))
    return dict(record) if record is not None else None


def _force_env_set() -> bool:
    for key in FORCE_ENV_KEYS:
        value = str(os.environ.get(key) or "").strip().casefold()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False


def execution_authorized(
    config: FrozenCanaryConfig,
    *,
    repo_root: Path | None = None,
    authorization_path: Path | None = None,
    want: str = "execution",
) -> bool:
    if _force_env_set():
        return False
    if config.config_hash == RETIRED_LEDGER_V5_HASH:
        return False
    record = authorization_record(
        config.config_hash,
        repo_root=repo_root,
        path=authorization_path,
    )
    if record is None:
        return False
    identity = dataset_identity(load_dataset_payload(repo_root=repo_root))
    stored_identity = record.get("dataset_identity")
    if not isinstance(stored_identity, Mapping):
        return False
    for key, value in identity.items():
        if stored_identity.get(key) != value:
            return False
    if record.get("suite") != SUITE or record.get("dataset_kind") != DATASET_KIND:
        return False
    if record.get("dataset_consumed") is True:
        return False
    ancestor = str(record.get("required_implementation_ancestor") or "").strip()
    if ancestor and repo_root is not None:
        head = git_snapshot(Path(repo_root))["lumenfin_commit"]
        if not _is_git_ancestor(Path(repo_root), ancestor, head):
            return False
    flag = {
        "execution": "execution_authorized",
        "preflight": "preflight_authorized",
        "remote": "remote_run_authorized",
    }[want]
    if not _strict_true(record.get(flag)):
        return False
    if want == "preflight":
        used = int(record.get("official_preflight_executions") or 0)
        maximum = int(record.get("max_official_preflight_executions") or 0)
        return used < maximum
    if want == "remote":
        used = int(record.get("official_remote_executions") or 0)
        maximum = int(record.get("max_official_remote_executions") or 0)
        return used < maximum
    return True


def _is_git_ancestor(repo_root: Path, ancestor: str, head: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, head],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0


def refuse_unauthorized(
    config: FrozenCanaryConfig,
    *,
    repo_root: Path | None = None,
    authorization_path: Path | None = None,
    output_dir: Path | None = None,
    preflight_output_dir: Path | None = None,
    want: str = "execution",
) -> None:
    if output_dir is not None:
        assert_isolated_path(output_dir, field="output-dir")
        if Path(output_dir).as_posix() != DEFAULT_OFFICIAL_OUTPUT_DIR.as_posix() and want != "test":
            if Path(output_dir).name != DEFAULT_OFFICIAL_OUTPUT_DIR.name:
                raise CanaryError("output directory swap is not authorized")
    if preflight_output_dir is not None:
        assert_isolated_path(preflight_output_dir, field="preflight-dir")
    if not execution_authorized(
        config,
        repo_root=repo_root,
        authorization_path=authorization_path,
        want=want,
    ):
        raise CanaryError("synthetic alias compliance is not authorized")


def parse_cli_guard(argv: list[str] | None = None) -> dict[str, Any]:
    args = list(argv or [])
    for item in args:
        key = item.split("=", 1)[0]
        if key in FORBIDDEN_CLI_FLAGS:
            raise CanaryError("synthetic alias compliance CLI refuses runtime overrides")
        lowered = item.casefold()
        if any(token in lowered for token in ("financebench", "public_holdout", "public-dev", "holdout", "ledger")):
            raise CanaryError("synthetic alias compliance CLI refuses reserved eval identities")
    return {"argv": args}


def resume_identity(
    config: FrozenCanaryConfig,
    *,
    repo_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    snapshot = git_snapshot(repo_root)
    identity = dataset_identity(load_dataset_payload(repo_root=repo_root))
    return {
        "commit": snapshot["lumenfin_commit"],
        "config_hash": config.config_hash,
        "dataset_sha256": identity["dataset_sha256"],
        "prompt_sha256": prompt_sha256(),
        "output_directory": Path(output_dir).as_posix(),
        "model": str(config.field("provider", "model") or DEFAULT_MODEL),
        "endpoint_sha256": str(config.field("provider", "chat_base_url_sha256") or chat_base_url_sha256()),
        "at_least_once": True,
        "exactly_once": False,
        "unobserved_inflight_possible": True,
    }


def assert_resume_compatible(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    for key in (
        "commit",
        "config_hash",
        "dataset_sha256",
        "prompt_sha256",
        "output_directory",
        "model",
        "endpoint_sha256",
    ):
        if existing.get(key) != expected.get(key):
            raise CanaryError("resume identity mismatch")


def assert_output_not_overwritten(path: Path, *, resume: bool = False) -> None:
    if not path.exists():
        return
    if path.is_file():
        raise CanaryError("refusing to overwrite an existing output file")
    if any(path.iterdir()) and not resume:
        raise CanaryError("refusing to overwrite a non-empty output directory")


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
            handle.write("\n")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def credential_presence() -> dict[str, Any]:
    raw = os.getenv("DEEPSEEK_API_KEY")
    return {"source": "env", "present": bool(raw and str(raw).strip())}


def assert_clean_worktree(repo_root: Path) -> dict[str, Any]:
    snapshot = git_snapshot(repo_root)
    if snapshot["worktree_dirty"]:
        raise CanaryError("official execution requires a clean worktree")
    return snapshot


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return float(ordered[index])


def build_preflight_payload(
    config: FrozenCanaryConfig,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    identity = dataset_identity(load_dataset_payload(repo_root=repo_root))
    snapshot = git_snapshot(repo_root)
    record = authorization_record(config.config_hash, repo_root=repo_root) or {}
    return {
        "kind": "preflight",
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "suite": SUITE,
        "dataset_kind": DATASET_KIND,
        "protocol": CITATION_ALIAS_PROTOCOL_VERSION,
        "config_hash": config.config_hash,
        "dataset_hash": identity["dataset_sha256"],
        "status": "PREFLIGHT_BUILT_NOT_AUTHORIZED",
        "executed_at": "",
        "exit_code": 2,
        "execution_commit": snapshot["lumenfin_commit"],
        "authorization_identity": {
            "authorization_sha256": (load_authorization(repo_root=repo_root) or {}).get(
                "authorization_sha256"
            ),
            "identity_status": record.get("identity_status"),
            "preflight_authorized": record.get("preflight_authorized"),
            "remote_run_authorized": record.get("remote_run_authorized"),
        },
        "cases_total": LOCKED_CASE_COUNT,
        "cases_remaining": LOCKED_CASE_COUNT,
        "cases_executed": 0,
        "remote_request_count": 0,
        "recorded_remote_calls": 0,
        "logical_generate_calls": 0,
        "public_holdout_used": False,
        "financebench_used": False,
        "ledger_public_dev_used": False,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "financial_accuracy_claim": False,
        "retrieval_quality_claim": False,
        "prompt_retuning_after_result": False,
        "gold_not_exposed_to_generator": True,
        "case_count": identity["case_count"],
        "dataset_identity": identity,
        "prompt_sha256": prompt_sha256(),
        "alias_protocol_version": CITATION_ALIAS_PROTOCOL_VERSION,
        "final_k": LOCKED_FINAL_K,
        "commit": snapshot["lumenfin_commit"],
        "gates": frozen_gate_thresholds(),
        "call_budget": config.field("call_budget"),
        "credential": credential_presence(),
        "billing_semantics": {
            "at_least_once": True,
            "exactly_once": False,
            "unobserved_inflight_possible": True,
        },
    }


def _assert_preflight_success_contract(report: Mapping[str, Any]) -> None:
    required = (
        "status",
        "schema_version",
        "executed_at",
        "exit_code",
        "execution_commit",
        "config_hash",
        "dataset_hash",
        "authorization_identity",
        "cases_total",
        "cases_remaining",
        "remote_request_count",
        "public_holdout_used",
        "financebench_used",
        "gold_not_exposed_to_generator",
    )
    missing = [key for key in required if key not in report]
    if missing:
        raise CanaryError("preflight report is missing required audit fields")
    if report.get("status") != "PREFLIGHT_OK":
        raise CanaryError("preflight status is not PREFLIGHT_OK")
    if report.get("exit_code") != 0:
        raise CanaryError("preflight success report must set exit_code 0")
    if report.get("cases_executed") != 0 or report.get("remote_request_count") != 0:
        raise CanaryError("preflight success cannot record cases or remote calls")
    if not str(report.get("executed_at") or ""):
        raise CanaryError("preflight success must record executed_at")
    if report.get("public_holdout_used") is not False or report.get("financebench_used") is not False:
        raise CanaryError("preflight success cannot record reserved eval use")


def load_official_preflight(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / DEFAULT_PREFLIGHT_OUTPUT_DIR / "preflight.json"
    if not path.is_file():
        raise CanaryError("successful official preflight is required")
    report = read_json_object(path, field="official-preflight")
    _assert_preflight_success_contract(report)
    return report


def run_preflight(
    *,
    repo_root: Path,
    frozen_config: FrozenCanaryConfig | None = None,
    authorization_path: Path | None = None,
    output_dir: Path | None = None,
    official: bool = True,
) -> dict[str, Any]:
    config = frozen_config or load_frozen_config(
        repo_root / DEFAULT_CONFIG_PATH,
        repo_root=repo_root,
        require_published=official,
    )
    dest = Path(output_dir or (repo_root / DEFAULT_PREFLIGHT_OUTPUT_DIR))
    if official:
        expected = (repo_root / DEFAULT_PREFLIGHT_OUTPUT_DIR).resolve()
        if dest.resolve() != expected:
            raise CanaryError("official preflight directory is fixed")
        refuse_unauthorized(
            config,
            repo_root=repo_root,
            authorization_path=authorization_path,
            preflight_output_dir=dest,
            want="preflight",
        )
        snapshot = assert_clean_worktree(repo_root)
        if not credential_presence()["present"]:
            raise CanaryError("DEEPSEEK_API_KEY is not present")
    else:
        snapshot = git_snapshot(repo_root)
    probe = NetworkProbe()
    probe.install()
    try:
        payload = build_preflight_payload(config, repo_root=repo_root)
        payload["status"] = "PREFLIGHT_OK"
        payload["exit_code"] = 0
        payload["executed_at"] = utc_now()
        payload["execution_commit"] = snapshot["lumenfin_commit"]
        payload["remote_request_count"] = probe.remote_request_count
        if payload["cases_executed"] != 0 or payload["remote_request_count"] != 0:
            raise CanaryError("preflight cannot execute cases or remote calls")
        _assert_preflight_success_contract(payload)
        assert_output_not_overwritten(dest)
        write_json_atomic(dest / "preflight.json", payload)
        return payload
    finally:
        probe.remove()
        if probe.remote_request_count:
            raise CanaryError("preflight made a remote call")


def generate_remote(bound: BoundCase, *, config: FrozenCanaryConfig) -> tuple[str, dict[str, Any]]:
    from ..llm import DeepSeekChatClient, LLMSettings

    settings = LLMSettings(
        api_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip() or None,
        base_url=DEFAULT_CHAT_BASE_URL,
        model=str(config.field("provider", "model") or DEFAULT_MODEL),
        timeout_seconds=float(config.field("provider", "timeout_seconds") or 60),
        max_retries=int(config.field("provider", "max_retries") or 2),
        retry_backoff_seconds=0.5,
    )
    if settings.model != DEFAULT_MODEL:
        raise CanaryError("live provider model is not the frozen DeepSeek model")
    if chat_base_url_sha256(settings.base_url) != chat_base_url_sha256():
        raise CanaryError("live provider endpoint hash drifted")
    if not settings.api_key:
        raise CanaryError("DEEPSEEK_API_KEY is not present")
    client = DeepSeekChatClient(settings)
    client.mark_usage_start()
    text = client.chat(
        SYNTHETIC_ALIAS_SYSTEM_PROMPT,
        bound.prompt_evidence,
        temperature=0.0,
        max_tokens=300,
    )
    return text, {
        "http_attempts": int(getattr(client, "last_attempts", 0) or 0),
        "prompt_tokens": int(client.usage_since_mark().get("prompt_tokens") or 0),
        "completion_tokens": int(client.usage_since_mark().get("completion_tokens") or 0),
    }


def _completed_case_ids(output_dir: Path) -> set[str]:
    completed: set[str] = set()
    cases_dir = output_dir / "cases"
    if cases_dir.is_dir():
        for path in cases_dir.glob("*.json"):
            completed.add(path.stem)
    jsonl = output_dir / "per_case.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, Mapping) and row.get("case_id"):
                completed.add(str(row["case_id"]))
    return completed


def _write_case_record(output_dir: Path, record: Mapping[str, Any]) -> None:
    case_id = str(record.get("case_id") or "")
    write_json_atomic(output_dir / "cases" / f"{case_id}.json", record)
    rows = []
    jsonl = output_dir / "per_case.jsonl"
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows = [row for row in rows if str(row.get("case_id") or "") != case_id]
    rows.append(dict(record))
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        "".join(canonical_dumps(row) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def run_remote_canary(
    *,
    repo_root: Path,
    config: FrozenCanaryConfig,
    resume: bool = False,
    generate_fn: Any = None,
) -> dict[str, Any]:
    dest = repo_root / DEFAULT_OFFICIAL_OUTPUT_DIR
    preflight = load_official_preflight(repo_root=repo_root)
    snapshot = assert_clean_worktree(repo_root)
    identity = dataset_identity(load_dataset_payload(repo_root=repo_root))
    if preflight.get("execution_commit") != snapshot["lumenfin_commit"]:
        raise CanaryError("official preflight is not bound to this HEAD")
    if preflight.get("config_hash") != config.config_hash:
        raise CanaryError("official preflight config hash mismatch")
    if preflight.get("dataset_hash") != identity["dataset_sha256"]:
        raise CanaryError("official preflight dataset hash mismatch")
    expected = resume_identity(config, repo_root=repo_root, output_dir=dest)
    if resume:
        manifest_path = dest / "resume_identity.json"
        if not manifest_path.is_file():
            raise CanaryError("resume requires a stored resume identity")
        assert_resume_compatible(read_json_object(manifest_path, field="resume-identity"), expected)
        if (dest / "resume_used.json").is_file():
            raise CanaryError("only one transport resume is authorized")
        write_json_atomic(dest / "resume_used.json", {"used": True, "at": utc_now()})
    else:
        assert_output_not_overwritten(dest)
        dest.mkdir(parents=True, exist_ok=True)
        write_json_atomic(dest / "resume_identity.json", expected)

    cases = load_cases(repo_root=repo_root)
    completed = _completed_case_ids(dest) if resume else set()
    totals = empty_metrics()
    totals["cases_total"] = LOCKED_CASE_COUNT
    latencies: list[float] = []
    case_rows: list[dict[str, Any]] = []
    recorded_remote = 0
    logical_calls = 0
    generate_attempts = 0
    provider_errors = 0

    for case in cases:
        if case.case_id in completed:
            case_path = dest / "cases" / f"{case.case_id}.json"
            if not case_path.is_file():
                raise CanaryError("resume is missing an atomic case record")
            existing = read_json_object(case_path, field="case")
            case_rows.append(existing)
            apply_case_metrics(totals, existing.get("metrics") or {})
            if existing.get("succeeded"):
                totals["cases_succeeded"] = int(totals["cases_succeeded"]) + 1
            else:
                totals["cases_failed"] = int(totals["cases_failed"]) + 1
            continue
        bound = bind_case(case)
        raw_text = ""
        usage = {"http_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0}
        started = time.perf_counter()
        try:
            if generate_fn is not None:
                raw_text = str(generate_fn(bound))
                usage["http_attempts"] = 1
            else:
                raw_text, usage = generate_remote(bound, config=config)
            logical_calls += 1
            recorded_remote += 1
            generate_attempts += int(usage.get("http_attempts") or 1)
        except Exception as exc:
            provider_errors += 1
            generate_attempts += int(usage.get("http_attempts") or 1)
            raw_text = ""
            scored = score_bound_output(bound, raw_text)
            scored["metrics"]["provider_error"] = redact_structured_error(str(exc))
        else:
            scored = score_bound_output(bound, raw_text)
        latency = time.perf_counter() - started
        latencies.append(latency)
        succeeded = (
            scored["metrics"]["citation_validation_failed"] is False
            and (
                scored["metrics"]["alias_mapping_success"]
                or scored["metrics"]["incomplete_case_handled"]
            )
        )
        record = {
            "case_id": case.case_id,
            "succeeded": succeeded,
            "latency_seconds": latency,
            "http_attempts": usage.get("http_attempts"),
            "metrics": scored["metrics"],
            "public_fields": scored["public_fields"],
            "finrun_validation": scored["metrics"].get("finrun_validation"),
        }
        _write_case_record(dest, record)
        case_rows.append(record)
        apply_case_metrics(totals, scored["metrics"])
        if succeeded:
            totals["cases_succeeded"] = int(totals["cases_succeeded"]) + 1
        else:
            totals["cases_failed"] = int(totals["cases_failed"]) + 1

    totals["logical_generate_calls"] = logical_calls
    totals["recorded_remote_calls"] = recorded_remote
    totals["generate_attempts"] = generate_attempts
    totals["provider_errors"] = provider_errors
    totals["latency_p50"] = _percentile(latencies, 50)
    totals["latency_p95"] = _percentile(latencies, 95)
    totals["protocol_gate_passed"] = evaluate_protocol_gate(totals)
    totals["synthetic_evidence_gate_passed"] = evaluate_synthetic_evidence_gate(totals)
    totals["release_gate_passed"] = evaluate_release_gate(totals)
    remaining = LOCKED_CASE_COUNT - len(case_rows)
    summary = {
        "kind": "official_remote_result",
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "status": "CANARY_COMPLETE" if remaining == 0 else "CANARY_INCOMPLETE",
        "executed_at": utc_now(),
        "exit_code": 0 if totals["release_gate_passed"] else 2,
        "execution_commit": snapshot["lumenfin_commit"],
        "config_hash": config.config_hash,
        "dataset_hash": identity["dataset_sha256"],
        "authorization_identity": preflight.get("authorization_identity"),
        "cases_total": LOCKED_CASE_COUNT,
        "cases_remaining": remaining,
        "remote_request_count": recorded_remote,
        "suite": SUITE,
        "dataset_kind": DATASET_KIND,
        "protocol": CITATION_ALIAS_PROTOCOL_VERSION,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "financial_accuracy_claim": False,
        "retrieval_quality_claim": False,
        "retuning_after_result_forbidden": True,
        "claim": "live-model synthetic alias protocol compliance",
        "public_holdout_used": False,
        "financebench_used": False,
        "ledger_public_dev_used": False,
        "gates": frozen_gate_thresholds(),
        "metrics": totals,
        "cases": [
            {
                "case_id": row["case_id"],
                "succeeded": row["succeeded"],
                "metrics": row["metrics"],
            }
            for row in case_rows
        ],
        "billing_semantics": {
            "at_least_once": True,
            "exactly_once": False,
            "unobserved_inflight_possible": True,
        },
    }
    write_json_atomic(dest / "summary.json", summary)
    return summary


def run_canary(
    *,
    repo_root: Path,
    frozen_config: FrozenCanaryConfig | None = None,
    authorization_path: Path | None = None,
    confirm_synthetic_alias_compliance: bool = False,
    allow_remote: bool = False,
    preflight_only: bool = False,
    resume: bool = False,
    official: bool = True,
    generate_fn: Any = None,
) -> dict[str, Any]:
    config = frozen_config or load_frozen_config(
        repo_root / DEFAULT_CONFIG_PATH,
        repo_root=repo_root,
        require_published=official,
    )
    if preflight_only and allow_remote:
        raise CanaryError("refusing --allow-remote with --preflight-only")
    if preflight_only and resume:
        raise CanaryError("refusing --resume with --preflight-only")
    if official:
        if preflight_only:
            refuse_unauthorized(
                config,
                repo_root=repo_root,
                authorization_path=authorization_path,
                want="preflight",
            )
            return run_preflight(
                repo_root=repo_root,
                frozen_config=config,
                authorization_path=authorization_path,
                official=True,
            )
        if not confirm_synthetic_alias_compliance or not allow_remote:
            raise CanaryError("remote canary requires --confirm-synthetic-alias-compliance and --allow-remote")
        refuse_unauthorized(
            config,
            repo_root=repo_root,
            authorization_path=authorization_path,
            output_dir=repo_root / DEFAULT_OFFICIAL_OUTPUT_DIR,
            want="remote",
        )
        record = authorization_record(config.config_hash, repo_root=repo_root) or {}
        if record.get("requires_successful_preflight") is not False:
            load_official_preflight(repo_root=repo_root)
        if resume and record.get("resume_authorized_only_for_transport_interruption") is True:
            pass
        return run_remote_canary(
            repo_root=repo_root,
            config=config,
            resume=resume,
            generate_fn=generate_fn,
        )
    if preflight_only:
        return run_preflight(
            repo_root=repo_root,
            frozen_config=config,
            authorization_path=authorization_path,
            official=False,
        )
    raise CanaryError("non-official remote canary is not implemented")


class NetworkProbe:
    """Count and block outbound socket connects."""

    def __init__(self) -> None:
        self.remote_request_count = 0
        self._installed = False
        self._orig_connect: Any = None
        self._orig_create: Any = None

    def install(self) -> None:
        if self._installed:
            return
        self._orig_connect = socket.socket.connect
        self._orig_create = socket.create_connection
        probe = self

        def connect(sock: socket.socket, *args: Any, **kwargs: Any) -> Any:
            probe.remote_request_count += 1
            raise OSError("synthetic alias compliance forbids network")

        def block(*_args: Any, **_kwargs: Any) -> Any:
            probe.remote_request_count += 1
            raise OSError("synthetic alias compliance forbids network")

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.create_connection = block  # type: ignore[assignment]
        self._installed = True

    def remove(self) -> None:
        if not self._installed:
            return
        if self._orig_connect is not None:
            socket.socket.connect = self._orig_connect  # type: ignore[method-assign]
        if self._orig_create is not None:
            socket.create_connection = self._orig_create  # type: ignore[assignment]
        self._installed = False

    def __enter__(self) -> "NetworkProbe":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()
