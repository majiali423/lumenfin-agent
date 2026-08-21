"""Offline synthetic structured-citation contract canary.

Call chain (production functions, not a parallel pipeline):

    chunk_document()
      → rag_evidence hits (same shape retrieval writes after Hybrid retrieve)
      → SynthesisMixin.claim_binder / build_claims / EvidenceRef.chunk_id
      → SynthesisMixin.synthesizer → _attach_structured_answer
      → build_structured_answer_from_state / validate_structured_answer
      → public_structured_answer_fields + AnalyzeResponse (API triple)
      → export_finrun_state
      → LEDGER parse_answer_payload / account_ledger_citations /
        citation_supported / score_generated_answer

This suite is a synthetic contract canary. It is not product accuracy, RAG
recall, FinanceBench, or a LEDGER benchmark score. It never opens
public_holdout and never rewrites sealed LEDGER aggregates.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..agents.runtime import AgentRuntime
from ..api.schemas import AnalyzeResponse
from ..claims import build_claims, filter_verified
from ..finrun import export_finrun_state
from ..knowledge_store import InMemoryKnowledgeStore
from ..llm import BaseLLMClient
from ..memory import ReasoningMemory, SessionMemory
from ..rag.chunking import chunk_document
from ..structured_answer import (
    CITATION_PATH_VALIDATION_FAILED,
    CITATION_SOURCE_LEGACY_STRUCTURED,
    CITATION_SOURCE_LEGACY_TEXT,
    CITATION_SOURCE_STRUCTURED,
    CITATION_SOURCE_UNAVAILABLE,
    CITATION_VALIDATION_FAILED,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    StructuredAnswerError,
    allowed_evidence_from_state,
    build_structured_answer_from_state,
    normalize_citations,
    public_structured_answer_fields,
    redact_structured_error,
    validate_structured_answer,
)
from .holdout.ledger_e2e import (
    account_ledger_citations,
    citation_supported,
    parse_answer_payload,
    score_generated_answer,
)

SUITE = "structured_citation_e2e_canary"
SUITE_VERSION = "1.0"
DEFAULT_RAW_OUTPUT_DIR = Path("outputs") / "structured_citation_canary_v1"
DEFAULT_SEAL_PATH = Path("data") / "eval_rag" / "structured_citation_canary_result.json"
FORBIDDEN_PATH_TOKENS = ("public_holdout", "financebench")
COMPANY = "CanaryCo"

CASE_IDS = (
    "A_single_verified_citation",
    "B_multi_citation_stable_order",
    "C_repair_stale_evidence",
    "D_incomplete_data",
    "E_cross_tenant",
    "F_cross_session",
    "G_unknown_chunk",
    "H_unverified_evidence",
    "I_conflicting_metadata",
    "J_legacy_structured",
    "K_unsupported_claim",
    "L_api_atomicity",
)

PRODUCTION_CALLS = (
    "chunk_document",
    "claim_binder",
    "synthesizer",
    "build_structured_answer_from_state",
    "public_structured_answer_fields",
    "export_finrun_state",
    "parse_answer_payload",
    "account_ledger_citations",
    "citation_supported",
)


class CanaryError(ValueError):
    """Fail-closed synthetic canary error (no secrets, no full filings)."""


def canonical_config() -> dict[str, Any]:
    return {
        "dataset_kind": "synthetic",
        "network_allowed": False,
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
    }


def canonical_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: Mapping[str, Any] | None = None) -> str:
    return sha256_text(canonical_dumps(config or canonical_config()))


def case_manifest() -> list[dict[str, str]]:
    return [{"case_id": case_id, "dataset_kind": "synthetic"} for case_id in CASE_IDS]


def case_manifest_hash() -> str:
    return sha256_text(canonical_dumps({"cases": case_manifest()}))


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


def refuse_forbidden_path(path: str | Path, *, field: str) -> Path:
    target = Path(path)
    lowered = str(target).replace("\\", "/").casefold()
    for token in FORBIDDEN_PATH_TOKENS:
        if token in lowered:
            raise CanaryError(f"{field} refuses {token}")
    return target


class NetworkProbe:
    """Count and block outbound socket connects. Git/subprocess stays local."""

    def __init__(self) -> None:
        self.remote_request_count = 0
        self._installed = False
        self._orig_connect: Callable[..., Any] | None = None
        self._orig_create: Callable[..., Any] | None = None

    def _block(self, *_args: Any, **_kwargs: Any) -> Any:
        self.remote_request_count += 1
        raise OSError("structured citation canary forbids network")

    def install(self) -> None:
        if self._installed:
            return
        self._orig_connect = socket.socket.connect
        self._orig_create = socket.create_connection
        probe = self

        def connect(sock: socket.socket, *args: Any, **kwargs: Any) -> Any:
            probe.remote_request_count += 1
            raise OSError("structured citation canary forbids network")

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.create_connection = self._block  # type: ignore[assignment]
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

    def __exit__(self, *_exc: object) -> None:
        self.remove()


class OfflineLLMClient(BaseLLMClient):
    backend_name = "canary-offline"
    model_name = "canary-offline"

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        raise CanaryError("structured citation canary forbids LLM chat")


class OfflineMarketClient:
    backend_name = "canary-offline"
    provider = "canary-offline"
    fallback_provider = "canary-offline"

    def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict[str, Any]:
        raise CanaryError("structured citation canary forbids market fetch")


@dataclass
class CallTrace:
    counts: dict[str, int] = field(default_factory=lambda: {name: 0 for name in PRODUCTION_CALLS})

    def mark(self, name: str) -> None:
        self.counts[name] = int(self.counts.get(name) or 0) + 1


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    detail: str
    citations: list[str] = field(default_factory=list)
    citation_source: str = CITATION_SOURCE_UNAVAILABLE
    structured_emitted: bool = False
    valid_citations: int = 0
    invalid_citations_detected: int = 0
    invalid_citations_missed: int = 0
    supported_claims: int = 0
    unsupported_claims_detected: int = 0
    api_atomicity_passed: bool = True
    finrun_roundtrip_passed: bool = True
    ledger_roundtrip_passed: bool = True
    production_calls: dict[str, int] = field(default_factory=dict)


def _offline_runtime() -> AgentRuntime:
    return AgentRuntime(
        session_memory=SessionMemory(),
        knowledge_memory=InMemoryKnowledgeStore(),
        reasoning_memory=ReasoningMemory(),
        llm_client=OfflineLLMClient(),
        market_data_client=OfflineMarketClient(),
        hybrid_retriever=None,
        rag_enabled=False,
        allow_sample_data=False,
        fetch_live_fundamentals=False,
        fetch_sec_fundamentals=False,
        data_mode="demo",
    )


def _fundamentals_payload() -> dict[str, Any]:
    return {
        "structured_source": "document_extracted",
        "market_data": {
            "revenue": 100.0,
            "ebitda": 50.0,
            "operating_income": 40.0,
            "r_and_d": 10.0,
        },
        "fundamentals_meta": {"fiscal_year": 2025},
        "fundamental_provenance": {
            key: {
                "source": "document_text",
                "confidence": "high",
                "period": "FY2025",
                "period_source": "document_text",
                "period_alignment": "exact",
                "citation": f"canary://{COMPANY}/{key}",
                "source_record_id": f"canary:{COMPANY}:FY2025:{key}",
            }
            for key in ("revenue", "ebitda", "operating_income", "r_and_d")
        },
        "source_documents": [],
    }


def _metrics() -> dict[str, float]:
    return {
        "ebitda_margin": 0.5,
        "operating_margin": 0.4,
        "r_and_d_intensity": 0.1,
    }


def mint_chunks(
    *,
    document_id: str,
    filename: str,
    pages: list[str],
    trace: CallTrace,
) -> list[dict[str, Any]]:
    trace.mark("chunk_document")
    return chunk_document(
        {
            "document_id": document_id,
            "filename": filename,
            "pages": pages,
            "issuer_companies": [COMPANY],
            "detected_companies": [COMPANY],
        }
    )


def hits_from_chunks(
    chunks: list[dict[str, Any]],
    *,
    tenant_id: str,
    session_id: str,
    extra: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for chunk in chunks:
        page = int(chunk.get("page") or 1)
        hit = {
            "chunk_id": str(chunk.get("chunk_id") or ""),
            "document_id": str(chunk.get("document_id") or ""),
            "filename": str(chunk.get("filename") or ""),
            "page": page,
            "text": str(chunk.get("text") or ""),
            "citation": f"{chunk.get('filename')}#p{page}",
            "source_type": "rag",
            "retrieval_method": "canary_synthetic_injection",
            "period": "FY2025",
            "period_source": "document_text",
            "period_alignment": "exact",
            "tenant_id": tenant_id,
            "session_id": session_id,
        }
        if extra:
            hit.update(dict(extra))
        hits.append(hit)
    return hits


def _base_state(
    *,
    tenant_id: str,
    session_id: str,
    run_id: str,
    hits: list[dict[str, Any]],
    query: str,
    fatal_gap: bool = False,
    pages: list[str] | None = None,
) -> dict[str, Any]:
    payload = _fundamentals_payload()
    if fatal_gap:
        payload = {
            "structured_source": "none",
            "market_data": {},
            "fundamentals_meta": {},
            "source_documents": [],
        }
    return {
        "query": query,
        "run_id": run_id,
        "thread_id": session_id,
        "tenant_id": tenant_id,
        "rag_tenant_id": tenant_id,
        "companies": [COMPANY],
        "workflow_status": "incomplete_data" if fatal_gap else "completed",
        "fatal_data_gap": fatal_gap,
        "data_gap_detail": "synthetic canary withheld fundamentals" if fatal_gap else "",
        "financial_metrics": {} if fatal_gap else {COMPANY: _metrics()},
        "retrieved_docs": {COMPANY: payload},
        "rag_evidence": {COMPANY: hits},
        "document_contexts": [
            {
                "document_id": (hits[0].get("document_id") if hits else "canary-empty"),
                "filename": (hits[0].get("filename") if hits else "empty.txt"),
                "pages": pages or [],
                "issuer_companies": [COMPANY],
            }
        ],
        "risk_scores": {COMPANY: {}},
        "sentiment_analysis": {COMPANY: {}},
        "peer_comparison": {},
        "market_snapshots": {},
        "company_profiles": {},
        "output_format": "table_summary",
        "llm_backend": "canary-offline",
        "data_mode": "demo",
        "audit_log": [],
    }


def bind_and_synthesize(state: dict[str, Any], *, trace: CallTrace) -> dict[str, Any]:
    runtime = _offline_runtime()
    trace.mark("claim_binder")
    binder_update = runtime.claim_binder(state)
    merged = {**state, **binder_update}
    trace.mark("synthesizer")
    synth_update = runtime.synthesizer(merged)
    out = {**merged, **synth_update}
    if out.get("structured_answer"):
        trace.mark("build_structured_answer_from_state")
    return out


def api_fields(result: Mapping[str, Any], *, trace: CallTrace) -> dict[str, Any] | None:
    trace.mark("public_structured_answer_fields")
    return public_structured_answer_fields(result)


def serialize_api(result: Mapping[str, Any], *, trace: CallTrace) -> dict[str, Any]:
    structured = api_fields(result, trace=trace)
    response = AnalyzeResponse(
        thread_id=str(result.get("thread_id") or ""),
        llm_backend=str(result.get("llm_backend") or "canary-offline"),
        workflow_status=str(result.get("workflow_status") or "completed"),
        clarification_questions=list(result.get("clarification_questions") or []),
        final_report=str(result.get("final_report") or ""),
        executive_summary=result.get("executive_summary"),
        compliance_summary=result.get("compliance_summary"),
        audit_log=list(result.get("audit_log") or []),
        artifacts={},
        state={},
        answer=None if structured is None else structured["answer"],
        citations=[] if structured is None else list(structured["citations"]),
        structured_answer_schema_version=(
            None if structured is None else structured["structured_answer_schema_version"]
        ),
    )
    return response.model_dump()


def export_finrun(state: Mapping[str, Any], *, trace: CallTrace) -> dict[str, Any]:
    trace.mark("export_finrun_state")
    return export_finrun_state(dict(state))


def ledger_from_structured(
    *,
    structured: Mapping[str, Any],
    hits: list[Mapping[str, Any]],
    qrels: Mapping[str, int],
    gold_value: float,
    tenant_id: str,
    session_id: str,
    trace: CallTrace,
    extra_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "answer": str(structured.get("answer") or ""),
        "citations": list(structured.get("citations") or []),
        "structured_answer_schema_version": structured.get("structured_answer_schema_version"),
        "citation_source": structured.get("citation_source"),
        "value": gold_value,
        "abstain": False,
    }
    if extra_payload:
        payload.update(dict(extra_payload))
    raw = json.dumps(payload, ensure_ascii=False)
    trace.mark("parse_answer_payload")
    parsed = parse_answer_payload(raw)
    parsed["tenant_id"] = tenant_id
    parsed["session_id"] = session_id
    trace.mark("account_ledger_citations")
    accounting = account_ledger_citations(
        cited_chunk_ids=list(parsed.get("citations") or []),
        hits=hits,
        qrels=qrels,
        citation_source=str(parsed.get("citation_source") or CITATION_SOURCE_UNAVAILABLE),
        tenant_id=tenant_id,
        session_id=session_id,
    )
    trace.mark("citation_supported")
    supported = citation_supported(list(parsed.get("citations") or []), hits, qrels)
    scored = score_generated_answer(
        gold_value=gold_value,
        parsed=parsed,
        hits=hits,
        qrels=qrels,
    )
    return {
        "parsed": parsed,
        "accounting": accounting,
        "supported": supported,
        "scored": scored,
    }


def _ids_match(*groups: list[str]) -> bool:
    first = list(groups[0])
    return all(list(group) == first for group in groups[1:])


def _qrels_for_hits(hits: list[Mapping[str, Any]]) -> dict[str, int]:
    return {str(hit.get("document_id") or ""): 1 for hit in hits if hit.get("document_id")}


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise CanaryError(detail)


def _case_a(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-a", "canary-session-a", "canary-run-a"
    pages = [
        (
            f"{COMPANY} FY2025 consolidated results. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    chunks = mint_chunks(document_id="canary-doc-a", filename="canary_a.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="What is CanaryCo revenue?", pages=pages),
        trace=trace,
    )
    structured = state.get("structured_answer") or {}
    api = serialize_api(state, trace=trace)
    finrun = export_finrun(state, trace=trace)
    fin_sa = finrun.get("structured_answer") or {}
    ledger = ledger_from_structured(
        structured=fin_sa,
        hits=hits,
        qrels=_qrels_for_hits(hits),
        gold_value=100.0,
        tenant_id=tenant,
        session_id=session,
        trace=trace,
    )
    citations = list(structured.get("citations") or [])
    _require(bool(citations), "A expected at least one verified citation")
    _require(all(any(hit["chunk_id"] == item for hit in hits) for item in citations), "A citation is not a minted chunk id")
    _require(
        _ids_match(citations, list(api.get("citations") or []), list(fin_sa.get("citations") or []), list(ledger["parsed"].get("citations") or [])),
        "A citation ids diverged across structured/API/FinRun/LEDGER",
    )
    _require(structured.get("citation_source") == CITATION_SOURCE_STRUCTURED, "A citation_source was not structured")
    _require(api.get("answer") == api.get("final_report"), "A API answer did not match final_report")
    _require(api.get("structured_answer_schema_version") == STRUCTURED_ANSWER_SCHEMA_VERSION, "A missing API schema version")
    _require(ledger["supported"] is True, "A LEDGER should treat gold document as supported")
    _require(fin_sa.get("citation_validation") != CITATION_VALIDATION_FAILED, "A FinRun unexpectedly degraded")
    return CaseResult(
        case_id="A_single_verified_citation",
        passed=True,
        detail="single verified chunk id identical across structured/API/FinRun/LEDGER",
        citations=citations,
        citation_source=str(structured.get("citation_source")),
        structured_emitted=True,
        valid_citations=len(citations),
        supported_claims=1,
        production_calls=dict(trace.counts),
    )


def _case_b(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-b", "canary-session-b", "canary-run-b"
    pages = [
        f"{COMPANY} revenue was 100.0 billion USD in FY2025.",
        f"{COMPANY} EBITDA was 50.0 billion USD in FY2025.",
        (
            f"{COMPANY} operating income was 40.0 billion USD in FY2025. "
            f"{COMPANY} R&D expense was 10.0 billion USD in FY2025."
        ),
    ]
    chunks = mint_chunks(document_id="canary-doc-b", filename="canary_b.txt", pages=pages, trace=trace)
    minted_hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    # Retrieval order is reversed so first-seen claim order cannot match hit order.
    hits = list(reversed(minted_hits))
    retrieval_ids = [str(hit["chunk_id"]) for hit in hits if hit.get("chunk_id")]
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="Compare CanaryCo metrics.", pages=pages),
        trace=trace,
    )
    structured = state.get("structured_answer") or {}
    citations = list(structured.get("citations") or [])
    _require(len(citations) >= 3, "B expected at least three unique citations")
    _require(citations != retrieval_ids[: len(citations)], "B citation order matched retrieval order")
    duplicated = citations + [citations[0], citations[-1]]
    deduped = normalize_citations(duplicated)
    _require(deduped == citations, "B first-seen dedupe did not preserve builder order")
    allowed = allowed_evidence_from_state(state)
    validate_structured_answer(
        {
            "answer": str(structured.get("answer") or "ok"),
            "citations": duplicated,
            "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
            "citation_source": CITATION_SOURCE_STRUCTURED,
            "workflow_status": "completed",
        },
        allowed=allowed,
        expected_tenant_id=tenant,
        expected_session_id=session,
        require_citation_for_factual=True,
    )
    api = serialize_api(state, trace=trace)
    finrun = export_finrun(state, trace=trace)
    ledger = ledger_from_structured(
        structured=finrun.get("structured_answer") or {},
        hits=hits,
        qrels=_qrels_for_hits(hits),
        gold_value=100.0,
        tenant_id=tenant,
        session_id=session,
        trace=trace,
    )
    _require(
        _ids_match(citations, list(api.get("citations") or []), list((finrun.get("structured_answer") or {}).get("citations") or [])),
        "B ids diverged after order/dedupe",
    )
    _require(ledger["supported"] is True, "B should remain supported")
    return CaseResult(
        case_id="B_multi_citation_stable_order",
        passed=True,
        detail="multi-cite first-seen order differs from retrieval and survives dedupe",
        citations=citations,
        citation_source=CITATION_SOURCE_STRUCTURED,
        structured_emitted=True,
        valid_citations=len(citations),
        supported_claims=1,
        production_calls=dict(trace.counts),
    )


def _case_c(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-c", "canary-session-c", "canary-run-c"
    old_pages = [f"{COMPANY} revenue was 12.0 billion USD in FY2024. Stale attempt 1."]
    new_pages = [
        (
            f"{COMPANY} FY2025 results. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    old_chunks = mint_chunks(document_id="canary-doc-c-attempt1", filename="canary_c_old.txt", pages=old_pages, trace=trace)
    new_chunks = mint_chunks(document_id="canary-doc-c-attempt2", filename="canary_c_new.txt", pages=new_pages, trace=trace)
    old_hits = hits_from_chunks(old_chunks, tenant_id=tenant, session_id=session, extra={"stale_repair_attempt": True})
    new_hits = hits_from_chunks(new_chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(
            tenant_id=tenant,
            session_id=session,
            run_id=run,
            hits=new_hits,
            query="Repair CanaryCo FY2025.",
            pages=new_pages,
        ),
        trace=trace,
    )
    structured = state.get("structured_answer") or {}
    citations = list(structured.get("citations") or [])
    old_ids = {str(hit["chunk_id"]) for hit in old_hits}
    new_ids = {str(hit["chunk_id"]) for hit in new_hits}
    _require(citations, "C expected attempt-2 citations")
    _require(all(item in new_ids for item in citations), "C cited a non-attempt-2 chunk")
    _require(not any(item in old_ids for item in citations), "C leaked attempt-1 chunk into success path")
    stale_payload = dict(structured)
    stale_payload["citations"] = [next(iter(old_ids))]
    detected = 0
    missed = 0
    try:
        validate_structured_answer(
            stale_payload,
            allowed=allowed_evidence_from_state({**state, "rag_evidence": {COMPANY: old_hits + new_hits}}),
            expected_tenant_id=tenant,
            expected_session_id=session,
            require_citation_for_factual=False,
        )
        missed += 1
    except StructuredAnswerError as exc:
        _require("stale" in str(exc).casefold() or "unknown" in str(exc).casefold(), "C missed stale/unknown wording")
        detected += 1
    finrun = export_finrun(state, trace=trace)
    serialize_api(state, trace=trace)
    ledger_from_structured(
        structured=finrun.get("structured_answer") or {},
        hits=new_hits,
        qrels=_qrels_for_hits(new_hits),
        gold_value=100.0,
        tenant_id=tenant,
        session_id=session,
        trace=trace,
    )
    _require(missed == 0, "C validator missed stale attempt-1 citation")
    return CaseResult(
        case_id="C_repair_stale_evidence",
        passed=True,
        detail="attempt-2 only; attempt-1 stale citation fail-closed",
        citations=citations,
        citation_source=CITATION_SOURCE_STRUCTURED,
        structured_emitted=True,
        valid_citations=len(citations),
        invalid_citations_detected=detected,
        invalid_citations_missed=missed,
        supported_claims=1,
        production_calls=dict(trace.counts),
    )


def _case_d(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-d", "canary-session-d", "canary-run-d"
    pages = [f"{COMPANY} qualitative outlook without extractable FY fundamentals."]
    chunks = mint_chunks(document_id="canary-doc-d", filename="canary_d.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(
            tenant_id=tenant,
            session_id=session,
            run_id=run,
            hits=hits,
            query="Compute CanaryCo EBITDA margin.",
            fatal_gap=True,
            pages=pages,
        ),
        trace=trace,
    )
    structured = state.get("structured_answer") or {}
    claims = filter_verified(build_claims(state))
    numeric = [claim for claim in claims if claim.claim_type in {"numeric", "growth"} and claim.value is not None]
    _require(state.get("workflow_status") == "incomplete_data", "D workflow_status was not incomplete_data")
    _require(not numeric, "D minted verified numeric claims without fundamentals")
    _require(list(structured.get("citations") or []) == [], "D citations should be empty")
    _require(structured.get("citation_source") == CITATION_SOURCE_UNAVAILABLE, "D must not look like structured success")
    _require(structured.get("citation_validation") != CITATION_VALIDATION_FAILED, "D empty citations are valid unavailable")
    api = serialize_api(state, trace=trace)
    _require(api.get("structured_answer_schema_version") == STRUCTURED_ANSWER_SCHEMA_VERSION, "D still emits a valid triple")
    finrun = export_finrun(state, trace=trace)
    fin_sa = finrun.get("structured_answer") or {}
    _require(fin_sa.get("citation_source") == CITATION_SOURCE_UNAVAILABLE, "D FinRun source was not unavailable")
    ledger = ledger_from_structured(
        structured={**fin_sa, "value": None, "abstain": True},
        hits=hits,
        qrels=_qrels_for_hits(hits),
        gold_value=100.0,
        tenant_id=tenant,
        session_id=session,
        trace=trace,
        extra_payload={"value": None, "abstain": True, "citations": []},
    )
    _require(ledger["parsed"].get("citation_source") == CITATION_SOURCE_UNAVAILABLE, "D LEDGER source was not unavailable")
    return CaseResult(
        case_id="D_incomplete_data",
        passed=True,
        detail="incomplete_data empty citations with unavailable source",
        citations=[],
        citation_source=CITATION_SOURCE_UNAVAILABLE,
        structured_emitted=True,
        production_calls=dict(trace.counts),
    )


def _mutate_scope(*, mismatch: str, trace: CallTrace) -> CaseResult:
    case_id = "E_cross_tenant" if mismatch == "tenant" else "F_cross_session"
    tenant, session, run = f"canary-tenant-{mismatch[0]}", f"canary-session-{mismatch[0]}", f"canary-run-{mismatch[0]}"
    pages = [
        (
            f"{COMPANY} FY2025. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    chunks = mint_chunks(document_id=f"canary-doc-{mismatch[0]}", filename=f"canary_{mismatch[0]}.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="CanaryCo revenue?", pages=pages),
        trace=trace,
    )
    mutated = deepcopy(state)
    if mismatch == "tenant":
        mutated["rag_tenant_id"] = tenant + "-other"
        mutated["tenant_id"] = tenant + "-other"
        expected_tenant, expected_session = tenant + "-other", session
    else:
        mutated["thread_id"] = session + "-other"
        mutated["run_id"] = session + "-other"
        expected_tenant, expected_session = tenant, session + "-other"
    detected = 0
    missed = 0
    try:
        validate_structured_answer(
            mutated.get("structured_answer") or {},
            allowed=allowed_evidence_from_state(mutated),
            expected_tenant_id=expected_tenant,
            expected_session_id=expected_session,
            require_citation_for_factual=False,
        )
        missed += 1
    except StructuredAnswerError:
        detected += 1
    finrun = export_finrun(mutated, trace=trace)
    fin_sa = finrun.get("structured_answer") or {}
    _require(fin_sa.get("citation_validation") == CITATION_VALIDATION_FAILED, f"{case_id} FinRun did not fail closed")
    _require(fin_sa.get("citation_path") == CITATION_PATH_VALIDATION_FAILED, f"{case_id} FinRun path was not validation_failed")
    api = serialize_api({**mutated, "structured_answer": fin_sa}, trace=trace)
    _require(api.get("structured_answer_schema_version") is None, f"{case_id} API leaked a half triple after FinRun degrade")
    _require(api.get("final_report"), f"{case_id} dropped final_report")
    _require(missed == 0, f"{case_id} validator missed the scope mismatch")
    serialize_api(state, trace=trace)
    export_finrun(state, trace=trace)
    return CaseResult(
        case_id=case_id,
        passed=True,
        detail=f"{mismatch} mismatch fail-closed",
        citations=list((state.get("structured_answer") or {}).get("citations") or []),
        citation_source=CITATION_SOURCE_STRUCTURED,
        structured_emitted=True,
        valid_citations=len(list((state.get("structured_answer") or {}).get("citations") or [])),
        invalid_citations_detected=detected,
        invalid_citations_missed=missed,
        api_atomicity_passed=True,
        finrun_roundtrip_passed=True,
        production_calls=dict(trace.counts),
    )


def _case_g(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-g", "canary-session-g", "canary-run-g"
    pages = [
        (
            f"{COMPANY} FY2025. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    chunks = mint_chunks(document_id="canary-doc-g", filename="canary_g.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="CanaryCo revenue?", pages=pages),
        trace=trace,
    )
    mutated = deepcopy(state)
    mutated["structured_answer"] = {
        **(state.get("structured_answer") or {}),
        "citations": ["forged-unknown-chunk"],
        "citation_source": CITATION_SOURCE_STRUCTURED,
    }
    detected = 0
    missed = 0
    try:
        validate_structured_answer(
            mutated["structured_answer"],
            allowed=allowed_evidence_from_state(state),
            expected_tenant_id=tenant,
            expected_session_id=session,
            require_citation_for_factual=False,
        )
        missed += 1
    except StructuredAnswerError as exc:
        _require("unknown" in str(exc).casefold(), "G did not report unknown chunk")
        detected += 1
    finrun = export_finrun(mutated, trace=trace)
    fin_sa = finrun.get("structured_answer") or {}
    _require(fin_sa.get("citation_validation") == CITATION_VALIDATION_FAILED, "G FinRun washed unknown into ordinary unavailable")
    _require(fin_sa.get("citation_path") == CITATION_PATH_VALIDATION_FAILED, "G FinRun missing validation_failed path")
    _require(list(fin_sa.get("citations") or []) == [], "G FinRun kept illegal citations")
    api = serialize_api({**mutated, "structured_answer": fin_sa}, trace=trace)
    _require(api.get("structured_answer_schema_version") is None, "G API emitted a success triple after FinRun degrade")
    _require(missed == 0, "G missed unknown chunk")
    return CaseResult(
        case_id="G_unknown_chunk",
        passed=True,
        detail="unknown chunk fail-closed; FinRun does not look like ordinary unavailable",
        invalid_citations_detected=detected,
        invalid_citations_missed=missed,
        structured_emitted=True,
        production_calls=dict(trace.counts),
    )


def _case_h(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-h", "canary-session-h", "canary-run-h"
    pages = [
        (
            f"{COMPANY} FY2025. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        ),
        "Methodology appendix without FY figures for this canary.",
    ]
    chunks = mint_chunks(document_id="canary-doc-h", filename="canary_h.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="CanaryCo revenue?", pages=pages),
        trace=trace,
    )
    verified = set((state.get("structured_answer") or {}).get("citations") or [])
    unverified = [str(hit["chunk_id"]) for hit in hits if hit["chunk_id"] not in verified]
    _require(unverified, "H needed an unbound chunk")
    mutated = deepcopy(state.get("structured_answer") or {})
    mutated["citations"] = [unverified[0]]
    detected = 0
    missed = 0
    try:
        validate_structured_answer(
            mutated,
            allowed=allowed_evidence_from_state(state),
            expected_tenant_id=tenant,
            expected_session_id=session,
            require_citation_for_factual=False,
        )
        missed += 1
    except StructuredAnswerError as exc:
        _require("unverified" in str(exc).casefold(), "H did not report unverified evidence")
        detected += 1
    serialize_api(state, trace=trace)
    export_finrun(state, trace=trace)
    _require(missed == 0, "H missed unverified citation")
    return CaseResult(
        case_id="H_unverified_evidence",
        passed=True,
        detail="unverified chunk fail-closed",
        citations=list(verified),
        citation_source=CITATION_SOURCE_STRUCTURED,
        structured_emitted=True,
        valid_citations=len(verified),
        invalid_citations_detected=detected,
        invalid_citations_missed=missed,
        production_calls=dict(trace.counts),
    )


def _case_i(trace: CallTrace) -> CaseResult:
    tenant, session = "canary-tenant-i", "canary-session-i"
    pages = [f"{COMPANY} revenue was 100.0 billion USD in FY2025."]
    chunks = mint_chunks(document_id="canary-doc-i", filename="canary_i.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    conflict = deepcopy(hits)
    if not conflict:
        raise CanaryError("I needed a minted chunk")
    conflict[0] = {
        **conflict[0],
        "tenant_id": tenant + "-other",
        "document_id": "canary-doc-i-other",
        "page": 99,
    }
    state = _base_state(
        tenant_id=tenant,
        session_id=session,
        run_id="canary-run-i",
        hits=hits + conflict,
        query="conflict",
        pages=pages,
    )
    detected = 0
    missed = 0
    try:
        allowed_evidence_from_state(state)
        missed += 1
    except StructuredAnswerError as exc:
        _require("conflict" in str(exc).casefold(), "I did not report metadata conflict")
        detected += 1
    _require(missed == 0, "I deduped conflicting metadata")
    return CaseResult(
        case_id="I_conflicting_metadata",
        passed=True,
        detail="same chunk id with conflicting tenant/document metadata fail-closed",
        invalid_citations_detected=detected,
        invalid_citations_missed=missed,
        production_calls=dict(trace.counts),
    )


def _case_j(trace: CallTrace) -> CaseResult:
    tenant, session = "canary-tenant-j", "canary-session-j"
    pages = [
        (
            f"{COMPANY} FY2025. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    chunks = mint_chunks(document_id="canary-doc-j", filename="canary_j.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    chunk_id = str(hits[0]["chunk_id"])
    raw = json.dumps(
        {"value": 100.0, "cited_chunk_ids": [chunk_id], "abstain": False},
        ensure_ascii=False,
    )
    trace.mark("parse_answer_payload")
    parsed = parse_answer_payload(raw)
    _require(parsed.get("citation_source") == CITATION_SOURCE_LEGACY_STRUCTURED, "J source was not legacy_structured")
    _require(parsed.get("citations") == [chunk_id], "J did not read cited_chunk_ids")
    trace.mark("account_ledger_citations")
    accounting = account_ledger_citations(
        cited_chunk_ids=list(parsed.get("citations") or []),
        hits=hits,
        qrels=_qrels_for_hits(hits),
        citation_source=str(parsed.get("citation_source")),
        tenant_id=tenant,
        session_id=session,
    )
    _require(accounting["legacy_fallback"] is True, "J accounting missed legacy_structured")
    _require(accounting["citation_source"] == CITATION_SOURCE_LEGACY_STRUCTURED, "J write-path source drifted")
    dumped = json.dumps({"parsed": parsed, "accounting": accounting}, ensure_ascii=False)
    _require(CITATION_SOURCE_LEGACY_TEXT not in dumped, "J writer emitted legacy_text")
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id="canary-run-j", hits=hits, query="legacy", pages=pages),
        trace=trace,
    )
    finrun = export_finrun(state, trace=trace)
    _require(CITATION_SOURCE_LEGACY_TEXT not in json.dumps(finrun, ensure_ascii=False), "J FinRun emitted legacy_text")
    return CaseResult(
        case_id="J_legacy_structured",
        passed=True,
        detail="cited_chunk_ids reads as legacy_structured; writers omit legacy_text",
        citations=[chunk_id],
        citation_source=CITATION_SOURCE_LEGACY_STRUCTURED,
        structured_emitted=True,
        valid_citations=1,
        supported_claims=1,
        ledger_roundtrip_passed=True,
        production_calls=dict(trace.counts),
    )


def _case_k(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-k", "canary-session-k", "canary-run-k"
    pages = [
        (
            f"{COMPANY} FY2025. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    chunks = mint_chunks(document_id="canary-doc-k", filename="canary_k.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="CanaryCo revenue?", pages=pages),
        trace=trace,
    )
    structured = state.get("structured_answer") or {}
    citations = list(structured.get("citations") or [])
    _require(citations, "K needed a legal verified citation")
    wrong_qrels = {str(hits[0].get("document_id") or "canary-doc-k"): 0, "gold-other-doc": 1}
    api = serialize_api(state, trace=trace)
    finrun = export_finrun(state, trace=trace)
    ledger = ledger_from_structured(
        structured=finrun.get("structured_answer") or {},
        hits=hits,
        qrels=wrong_qrels,
        gold_value=100.0,
        tenant_id=tenant,
        session_id=session,
        trace=trace,
    )
    _require(ledger["supported"] is False, "K treated a gold-mismatch as supported")
    _require(ledger["accounting"].get("unsupported_claim") is True, "K missed unsupported_claim")
    _require(ledger["accounting"].get("valid_citation", 0) >= 1, "K should still see a valid id")
    _require(api.get("citations") == citations, "K API ids drifted")
    return CaseResult(
        case_id="K_unsupported_claim",
        passed=True,
        detail="legal verified id that misses gold evidence is unsupported, not a pass",
        citations=citations,
        citation_source=CITATION_SOURCE_STRUCTURED,
        structured_emitted=True,
        valid_citations=len(citations),
        unsupported_claims_detected=1,
        production_calls=dict(trace.counts),
    )


def _case_l(trace: CallTrace) -> CaseResult:
    tenant, session, run = "canary-tenant-l", "canary-session-l", "canary-run-l"
    pages = [
        (
            f"{COMPANY} FY2025. {COMPANY} revenue was 100.0 billion USD. "
            f"{COMPANY} EBITDA was 50.0 billion USD. {COMPANY} operating income was 40.0 billion USD. "
            f"{COMPANY} R&D expense was 10.0 billion USD."
        )
    ]
    chunks = mint_chunks(document_id="canary-doc-l", filename="canary_l.txt", pages=pages, trace=trace)
    hits = hits_from_chunks(chunks, tenant_id=tenant, session_id=session)
    state = bind_and_synthesize(
        _base_state(tenant_id=tenant, session_id=session, run_id=run, hits=hits, query="CanaryCo revenue?", pages=pages),
        trace=trace,
    )
    full = serialize_api(state, trace=trace)
    _require(full.get("answer") and full.get("citations") and full.get("structured_answer_schema_version"), "L full triple missing")
    _require(full.get("final_report"), "L dropped final_report on success")
    _require(full.get("answer") == full.get("final_report"), "L answer/final_report diverged")

    missing = dict(state)
    missing.pop("structured_answer", None)
    no_sa = serialize_api(missing, trace=trace)
    _require(no_sa.get("answer") is None, "L missing structured still set answer")
    _require(no_sa.get("citations") == [], "L missing structured leaked citations")
    _require(no_sa.get("structured_answer_schema_version") is None, "L missing structured leaked version")
    _require(no_sa.get("final_report"), "L missing structured dropped final_report")

    failed_state = deepcopy(state)
    failed_state["structured_answer"] = {
        **(state.get("structured_answer") or {}),
        "citation_validation": CITATION_VALIDATION_FAILED,
        "validation_error": "synthetic",
    }
    failed = serialize_api(failed_state, trace=trace)
    _require(failed.get("answer") is None, "L failed validation leaked answer")
    _require(failed.get("structured_answer_schema_version") is None, "L failed validation leaked version")
    _require(failed.get("final_report"), "L failed validation dropped final_report")

    half = public_structured_answer_fields(
        {
            "final_report": "compat",
            "structured_answer": {
                "answer": "compat",
                "citations": [],
            },
        }
    )
    trace.mark("public_structured_answer_fields")
    _require(half is None, "L accepted a half triple without schema version")
    export_finrun(state, trace=trace)
    return CaseResult(
        case_id="L_api_atomicity",
        passed=True,
        detail="API triple is all-or-nothing; final_report remains",
        citations=list(full.get("citations") or []),
        citation_source=CITATION_SOURCE_STRUCTURED,
        structured_emitted=True,
        valid_citations=len(list(full.get("citations") or [])),
        api_atomicity_passed=True,
        production_calls=dict(trace.counts),
    )


CASE_RUNNERS: dict[str, Callable[[CallTrace], CaseResult]] = {
    "A_single_verified_citation": _case_a,
    "B_multi_citation_stable_order": _case_b,
    "C_repair_stale_evidence": _case_c,
    "D_incomplete_data": _case_d,
    "E_cross_tenant": lambda trace: _mutate_scope(mismatch="tenant", trace=trace),
    "F_cross_session": lambda trace: _mutate_scope(mismatch="session", trace=trace),
    "G_unknown_chunk": _case_g,
    "H_unverified_evidence": _case_h,
    "I_conflicting_metadata": _case_i,
    "J_legacy_structured": _case_j,
    "K_unsupported_claim": _case_k,
    "L_api_atomicity": _case_l,
}


def _case_to_dict(result: CaseResult) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "passed": result.passed,
        "detail": result.detail,
        "citation_count": len(result.citations),
        "citation_source": result.citation_source,
        "structured_emitted": result.structured_emitted,
        "valid_citations": result.valid_citations,
        "invalid_citations_detected": result.invalid_citations_detected,
        "invalid_citations_missed": result.invalid_citations_missed,
        "supported_claims": result.supported_claims,
        "unsupported_claims_detected": result.unsupported_claims_detected,
        "api_atomicity_passed": result.api_atomicity_passed,
        "finrun_roundtrip_passed": result.finrun_roundtrip_passed,
        "ledger_roundtrip_passed": result.ledger_roundtrip_passed,
        "production_calls": result.production_calls,
    }


def aggregate_metrics(results: list[CaseResult], *, remote_request_count: int) -> dict[str, Any]:
    failed = [item for item in results if not item.passed]
    return {
        "cases_total": len(results),
        "cases_passed": len(results) - len(failed),
        "cases_failed": len(failed),
        "structured_cases": sum(1 for item in results if item.structured_emitted),
        "structured_emitted": sum(1 for item in results if item.structured_emitted),
        "valid_citations": sum(item.valid_citations for item in results),
        "invalid_citations_detected": sum(item.invalid_citations_detected for item in results),
        "invalid_citations_missed": sum(item.invalid_citations_missed for item in results),
        "supported_claims": sum(item.supported_claims for item in results),
        "unsupported_claims_detected": sum(item.unsupported_claims_detected for item in results),
        "api_atomicity_passed": all(item.api_atomicity_passed for item in results),
        "finrun_roundtrip_passed": all(item.finrun_roundtrip_passed for item in results),
        "ledger_roundtrip_passed": all(item.ledger_roundtrip_passed for item in results),
        "remote_request_count": int(remote_request_count),
    }


def gates_passed(metrics: Mapping[str, Any]) -> bool:
    return (
        int(metrics.get("cases_failed") or 0) == 0
        and int(metrics.get("invalid_citations_missed") or 0) == 0
        and bool(metrics.get("api_atomicity_passed"))
        and bool(metrics.get("finrun_roundtrip_passed"))
        and bool(metrics.get("ledger_roundtrip_passed"))
        and int(metrics.get("remote_request_count") or 0) == 0
    )


def claim_flags() -> dict[str, Any]:
    return {
        "synthetic_contract_canary": True,
        "product_accuracy_claim": False,
        "retrieval_quality_claim": False,
        "public_holdout_used": False,
        "sealed_aggregate_modified": False,
    }


def empty_metrics() -> dict[str, Any]:
    return aggregate_metrics([], remote_request_count=0)


def run_cases(*, probe: NetworkProbe | None = None) -> tuple[list[CaseResult], CallTrace]:
    results: list[CaseResult] = []
    combined = CallTrace()
    for case_id in CASE_IDS:
        local = CallTrace()
        try:
            result = CASE_RUNNERS[case_id](local)
            result.production_calls = dict(local.counts)
            results.append(result)
        except Exception as exc:
            results.append(
                CaseResult(
                    case_id=case_id,
                    passed=False,
                    detail=redact_structured_error(str(exc)),
                    production_calls=dict(local.counts),
                )
            )
        for name, count in local.counts.items():
            combined.counts[name] = int(combined.counts.get(name) or 0) + int(count)
        if probe is not None and probe.remote_request_count:
            break
    return results, combined


def slim_result(
    *,
    metrics: Mapping[str, Any],
    results: list[CaseResult],
    git: Mapping[str, Any],
    raw_artifacts: list[dict[str, str]],
    executed_at: str,
    remote_request_count: int,
    worktree_dirty_allowed: bool,
) -> dict[str, Any]:
    payload = {
        **claim_flags(),
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "config": canonical_config(),
        "config_hash": config_hash(),
        "git": dict(git),
        "worktree_dirty_allowed": bool(worktree_dirty_allowed),
        "executed_at": executed_at,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "case_ids": list(CASE_IDS),
        "case_manifest_hash": case_manifest_hash(),
        "metrics": dict(metrics),
        "gates_passed": gates_passed(metrics),
        "case_results": [_case_to_dict(item) for item in results],
        "raw_artifacts": raw_artifacts,
        "remote_request_count": int(remote_request_count),
        "financebench_runs": 0,
        "public_holdout_reads": 0,
        "sealed_ledger_aggregate_writes": 0,
        "api_coverage": (
            "public_structured_answer_fields + AnalyzeResponse "
            "(same triple assignment as create_app._to_response); HTTP TestClient not used"
        ),
        "stubs": ["LLM chat", "market fetch", "embedding", "reranker", "external DB", "network sockets"],
    }
    return payload


def prepare_output_dir(path: Path) -> Path:
    refuse_forbidden_path(path, field="output-dir")
    if path.exists():
        if not path.is_dir():
            raise CanaryError("output-dir exists and is not a directory")
        if any(path.iterdir()):
            raise CanaryError("refusing to overwrite a non-empty output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    refuse_forbidden_path(path, field="output-path")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return sha256_file(path)


def write_invalid(path: Path, *, error: str, git: Mapping[str, Any] | None = None) -> None:
    payload = {
        **claim_flags(),
        "ok": False,
        "error": redact_structured_error(error),
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "config_hash": config_hash(),
        "git": dict(git or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_canary(
    *,
    output_dir: str | Path,
    repo_root: str | Path,
    require_clean_worktree: bool = True,
    seal_path: str | Path | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    cfg = dict(config or canonical_config())
    if cfg != canonical_config():
        raise CanaryError("config mismatch; synthetic canary is locked")
    if cfg.get("network_allowed") is not False:
        raise CanaryError("structured citation canary refuses network")
    if str(cfg.get("dataset_kind") or "") != "synthetic":
        raise CanaryError("structured citation canary only allows synthetic dataset_kind")
    git = git_snapshot(root)
    if require_clean_worktree and git.get("worktree_dirty"):
        raise CanaryError("structured citation canary requires a clean git worktree")
    out = prepare_output_dir(Path(output_dir))
    seal = Path(seal_path) if seal_path else None
    if seal is not None:
        refuse_forbidden_path(seal, field="seal-path")
        if seal.exists():
            raise CanaryError("refusing to overwrite an existing seal path")

    probe = NetworkProbe()
    executed_at = datetime.now(timezone.utc).isoformat()
    try:
        probe.install()
        results, _trace = run_cases(probe=probe)
        metrics = aggregate_metrics(results, remote_request_count=probe.remote_request_count)
        if probe.remote_request_count:
            raise CanaryError("structured citation canary observed a remote request")
        raw_cases = write_json(out / "cases.json", {"cases": [_case_to_dict(item) for item in results]})
        raw_metrics = write_json(out / "metrics.json", metrics)
        slim = slim_result(
            metrics=metrics,
            results=results,
            git=git,
            raw_artifacts=[
                {"name": "cases.json", "sha256": raw_cases},
                {"name": "metrics.json", "sha256": raw_metrics},
            ],
            executed_at=executed_at,
            remote_request_count=probe.remote_request_count,
            worktree_dirty_allowed=not require_clean_worktree,
        )
        write_json(out / "summary.json", slim)
        slim["raw_artifacts"] = [
            {"name": "cases.json", "sha256": raw_cases},
            {"name": "metrics.json", "sha256": raw_metrics},
            {"name": "summary.json", "sha256": sha256_file(out / "summary.json")},
        ]
        if not gates_passed(metrics):
            write_invalid(out / "invalid.json", error="canary gates failed", git=git)
            raise CanaryError("structured citation canary gates failed")
        if seal is not None:
            seal.parent.mkdir(parents=True, exist_ok=True)
            write_json(seal, slim)
        slim["ok"] = True
        return slim
    except Exception as exc:
        write_invalid(out / "invalid.json", error=str(exc), git=git)
        raise
    finally:
        probe.remove()


def parse_cli_guard(argv: list[str]) -> None:
    joined = " ".join(argv).casefold()
    if "--allow-remote" in joined:
        raise CanaryError("structured citation canary refuses --allow-remote")
    if "public_holdout" in joined:
        raise CanaryError("structured citation canary refuses public_holdout")
    if "financebench" in joined:
        raise CanaryError("structured citation canary refuses FinanceBench paths")
    if os.environ.get("LUMENFIN_ALLOW_REMOTE", "").strip() in {"1", "true", "yes"}:
        raise CanaryError("structured citation canary refuses LUMENFIN_ALLOW_REMOTE")
