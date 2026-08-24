"""Frozen LEDGER public_holdout E2E contract. Default-deny. Does not open holdout text."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from ..citation_alias import (
    LOCKED_FINAL_K,
    CitationAliasError,
    build_citation_alias_map,
    build_final_evidence_window,
    official_ranked_hits,
    parse_and_map_citation_aliases,
    render_prompt_evidence,
)
from ..structured_answer import (
    AllowedEvidence,
    CITATION_SOURCE_STRUCTURED,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    StructuredAnswerError,
    validate_structured_answer,
)
from .holdout.ledger_e2e import citation_supported, numeric_match

SUITE = "ledger_public_holdout_held_out_e2e"
CLAIM = "LEDGER public_holdout held-out end-to-end verified task success"
PRODUCT_TAG = "v0.1.0-rc.5"
PRODUCT_COMMIT = "31e8680aa89636f1fd897d7aa5ed7ca86317bd73"
CONTRACT_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_e2e_contract.json"
AUTHORIZATION_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_e2e_authorization.json"
RESULT_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_e2e_result.json"
PREFLIGHT_DIR = Path("outputs") / "ledger_public_holdout_e2e_preflight_v1"
OFFICIAL_DIR = Path("outputs") / "ledger_public_holdout_e2e_v1"
BLOCK_REASON = "no_compatible_prebuilt_index"
HASH_SKIP = frozenset({"config_hash", "authorization_sha256"})
FORBIDDEN_CLI = {
    "--dataset",
    "--dataset-path",
    "--parquet",
    "--parquet-path",
    "--split",
    "--limit",
    "--model",
    "--prompt",
    "--final-k",
    "--final_k",
    "--output-dir",
    "--preflight-dir",
    "--embedding",
    "--reranker",
    "--ignore-qrels",
    "--skip-citation-validation",
}
HOLDOUT_TOKENS = ("public_holdout", "public-holdout")


class HoldoutE2EError(ValueError):
    """Safe evaluation-governance error. No holdout text, gold, or secrets."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def canonical_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key not in HASH_SKIP}
    return sha256_text(canonical_dumps(body))


def wilson_interval(successes: int, n: int, *, z: float = 1.96) -> dict[str, Any]:
    if n <= 0:
        return {"n": 0, "successes": successes, "rate": None, "low": None, "high": None}
    if successes < 0 or successes > n:
        raise HoldoutE2EError("Wilson interval successes are out of range")
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))
    return {
        "n": n,
        "successes": successes,
        "rate": p,
        "low": max(0.0, center - margin),
        "high": min(1.0, center + margin),
        "method": "wilson_95",
    }


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutE2EError("JSON object required")
    return payload


def load_contract(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_PATH
    payload = read_json(path)
    if payload.get("schema_version") != "ledger_public_holdout_e2e_contract.v1":
        raise HoldoutE2EError("contract schema mismatch")
    if payload.get("claim") != CLAIM:
        raise HoldoutE2EError("contract claim mismatch")
    if payload.get("llm_judge_as_primary") is not False:
        raise HoldoutE2EError("LLM judge cannot be the primary metric")
    product = payload.get("product") or {}
    if product.get("product_commit") != PRODUCT_COMMIT or product.get("release_tag") != PRODUCT_TAG:
        raise HoldoutE2EError("contract is not locked to v0.1.0-rc.5")
    computed = compute_config_hash(payload)
    stored = str(payload.get("config_hash") or "")
    if stored and stored != computed:
        raise HoldoutE2EError("contract config_hash mismatch")
    payload["config_hash"] = computed
    return payload


def load_authorization(*, repo_root: Path) -> dict[str, Any]:
    payload = read_json(repo_root / AUTHORIZATION_PATH)
    if payload.get("kind") != "ledger_public_holdout_e2e_authorization":
        raise HoldoutE2EError("authorization kind mismatch")
    if payload.get("default_deny") is not True:
        raise HoldoutE2EError("authorization must be default-deny")
    return payload


def execution_authorized(contract: Mapping[str, Any], *, repo_root: Path, want: str) -> bool:
    auth = load_authorization(repo_root=repo_root)
    record = (auth.get("records") or {}).get(str(contract["config_hash"]))
    if not isinstance(record, Mapping):
        return False
    if record.get("dataset_consumed") is True or record.get("holdout_consumed") is True:
        return False
    if record.get("reason") == BLOCK_REASON:
        return False
    flags = {
        "preflight": "preflight_authorized",
        "remote": "remote_run_authorized",
    }
    flag = flags.get(want)
    if flag is None:
        return False
    return (
        record.get("execution_authorized") is True
        and record.get(flag) is True
        and record.get("identity_status") == "ONE_SHOT_AUTHORIZED"
    )


def refuse_unauthorized(*, repo_root: Path, argv: list[str] | None = None) -> None:
    args = list(argv or [])
    lowered = " ".join(args).casefold()
    for token in HOLDOUT_TOKENS:
        if token in lowered and "--confirm-public-holdout-consumption" not in args:
            raise HoldoutE2EError("CLI refuses holdout path or split overrides")
    for flag in FORBIDDEN_CLI:
        if flag in args:
            raise HoldoutE2EError("CLI refuses runtime overrides")
    contract = load_contract(repo_root=repo_root)
    if contract["index_gate"]["compatible_prebuilt_index_present"] is not False:
        raise HoldoutE2EError("index gate must stay blocked until a sealed prebuilt index exists")
    if execution_authorized(contract, repo_root=repo_root, want="preflight"):
        raise HoldoutE2EError("blocked contract cannot be authorized")
    if execution_authorized(contract, repo_root=repo_root, want="remote"):
        raise HoldoutE2EError("blocked contract cannot be authorized")
    raise HoldoutE2EError(
        "LEDGER public_holdout E2E is blocked: no compatible prebuilt index; "
        "document re-embedding is forbidden"
    )


def blocked_preflight_payload(contract: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "PREFLIGHT_BLOCKED",
        "schema_version": "ledger_public_holdout_e2e_preflight.v1",
        "exit_code": 2,
        "cases_executed": 0,
        "cases_total": 0,
        "remote_request_count": 0,
        "holdout_content_logged": False,
        "holdout_consumed": False,
        "product_commit": PRODUCT_COMMIT,
        "config_hash": contract["config_hash"],
        "block_reason": BLOCK_REASON,
        "document_reembedding_calls": 0,
        "public_dev_candidate_cache_used": False,
    }


def qrels_status(qrels: Mapping[str, int] | None) -> str:
    if not qrels:
        return "NOT_EVALUABLE"
    if not any(int(value) > 0 for value in qrels.values()):
        return "NOT_EVALUABLE"
    return "BOUND"


def evaluate_verified_e2e_case(
    *,
    predicted: float | None,
    gold: float | None,
    abstain: bool,
    citations: list[str],
    hits: list[Mapping[str, Any]],
    qrels: Mapping[str, int] | None,
    provider_success: bool,
    structured_contract_valid: bool,
    incomplete_data: bool = False,
) -> dict[str, Any]:
    support_status = qrels_status(qrels)
    if gold is None and not incomplete_data:
        answer_status = "NOT_EVALUABLE"
        answer_correct = False
    elif incomplete_data:
        answer_status = "INCOMPLETE_DATA"
        answer_correct = bool(abstain and predicted is None)
    else:
        answer_status = "EVALUABLE"
        answer_correct = bool(
            not abstain
            and numeric_match(predicted, float(gold))["matched"]
        )
    citation_ok = False
    if support_status == "BOUND":
        citation_ok = bool(citations) and citation_supported(citations, hits, qrels or {})
    verified = bool(
        provider_success
        and structured_contract_valid
        and answer_status == "EVALUABLE"
        and answer_correct
        and support_status == "BOUND"
        and citation_ok
    )
    return {
        "verified_e2e_success": verified,
        "answer_correct": answer_correct,
        "answer_metric_status": answer_status,
        "structured_contract_valid": structured_contract_valid,
        "citation_supported": citation_ok,
        "support_metric_valid": support_status == "BOUND",
        "qrels_bound": support_status == "BOUND",
        "qrels_status": support_status,
        "provider_success": provider_success,
        "incomplete_data": incomplete_data,
    }


def summarize_cases(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    successes = sum(1 for row in rows if row.get("verified_e2e_success") is True)
    evaluable = [
        row
        for row in rows
        if row.get("answer_metric_status") == "EVALUABLE"
        and row.get("qrels_status") == "BOUND"
        and row.get("provider_success") is True
    ]
    conditional_successes = sum(1 for row in evaluable if row.get("verified_e2e_success") is True)
    return {
        "claim": CLAIM,
        "dataset_specific": True,
        "held_out": True,
        "single_use": True,
        "general_product_accuracy_claim": False,
        "strict_verified_e2e_success": wilson_interval(successes, total),
        "evaluable_coverage": (len(evaluable) / total) if total else None,
        "conditional_verified_e2e_success": wilson_interval(
            conditional_successes, len(evaluable)
        ),
        "cases_total": total,
        "evaluable_cases": len(evaluable),
    }


def fixture_hits() -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": f"holdout-fix:doc:p1:c{index}",
            "document_id": f"NYSE_FIX_2023/page_{index:04d}",
            "text": f"workshop crate count {index}",
            "origin": "test_fixture",
            "tenant_id": "t1",
            "session_id": "s1",
        }
        for index in range(10)
    ]


def apply_alias_and_validate(
    raw_citations: list[str],
    hits: list[Mapping[str, Any]],
    *,
    tenant_id: str = "t1",
    session_id: str = "s1",
) -> dict[str, Any]:
    ranked = official_ranked_hits(hits, origin="test_fixture")
    window = build_final_evidence_window(ranked, final_k=LOCKED_FINAL_K)
    alias_map = build_citation_alias_map(
        window,
        case_id="FIX",
        attempt_id="1",
        tenant_id=tenant_id,
        session_id=session_id,
    )
    prompt = render_prompt_evidence(window, alias_map, query_text="fixture")
    if any(str(hit["chunk_id"]) in prompt for hit in window):
        raise HoldoutE2EError("stable chunk IDs leaked into the fixture prompt")
    try:
        mapped = parse_and_map_citation_aliases(
            raw_citations,
            alias_map,
            expected_case_id="FIX",
            expected_attempt_id="1",
            expected_tenant_id=tenant_id,
            expected_session_id=session_id,
        )
        unknown = 0
        raw_leak = 0
        citations = list(mapped)
    except CitationAliasError as exc:
        message = str(exc).casefold()
        unknown = int("unknown" in message or "allowlist" in message or "syntax" in message)
        raw_leak = int("raw" in message or "chunk" in message or "stable" in message)
        citations = []
        if not unknown and not raw_leak:
            unknown = 1
    allowed = [
        AllowedEvidence(
            chunk_id=str(hit["chunk_id"]),
            tenant_id=tenant_id,
            session_id=session_id,
            verified=not bool(hit.get("unverified")),
            stale=bool(hit.get("stale_repair_attempt")),
        )
        for hit in window
    ]
    structured_ok = False
    validation_failed = False
    if citations:
        try:
            validate_structured_answer(
                {
                    "answer": "ok",
                    "citations": citations,
                    "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                    "citation_source": CITATION_SOURCE_STRUCTURED,
                    "workflow_status": "completed",
                    "has_factual_conclusion": False,
                },
                allowed=allowed,
                expected_tenant_id=tenant_id,
                expected_session_id=session_id,
                require_citation_for_factual=False,
            )
            structured_ok = True
        except StructuredAnswerError:
            validation_failed = True
    return {
        "citations": citations,
        "unknown_aliases": unknown,
        "raw_chunk_id_leakage": raw_leak,
        "structured_contract_valid": structured_ok and not validation_failed,
        "citation_validation_failed": validation_failed,
        "hits": list(window),
    }
