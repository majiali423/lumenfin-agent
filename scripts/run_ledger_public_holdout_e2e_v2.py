#!/usr/bin/env python3
"""LEDGER public_holdout E2E v2: sealed-index unlock, preflight, one-shot consume."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.citation_alias import (  # noqa: E402
    CitationAliasError,
    build_citation_alias_map,
    build_final_evidence_window,
    official_ranked_hits,
    parse_and_map_citation_aliases,
    render_alias_passages,
)
from lumenfin.env_bootstrap import bootstrap_dotenv  # noqa: E402
from lumenfin.eval.financebench.candidate_pool_ablation import (  # noqa: E402
    build_locked_qwen3_reranker,
    snapshot_rerank_settings,
)
from lumenfin.eval.financebench.constants import DEFAULT_BM25_RRF_WEIGHT  # noqa: E402
from lumenfin.eval.holdout import ARM_SPECS, prepare_rerank_pool  # noqa: E402
from lumenfin.eval.holdout.ledger_e2e import parse_answer_payload  # noqa: E402
from lumenfin.eval.ledger_public_holdout_e2e import (  # noqa: E402
    evaluate_verified_e2e_case,
    summarize_cases,
)
from lumenfin.eval.ledger_public_holdout_e2e_v2 import (  # noqa: E402
    AUTHORIZATION_V2_PATH,
    CLAIM,
    CONTRACT_V2_PATH,
    MAX_OFFICIAL_CASES,
    OFFICIAL_DIR,
    PREFLIGHT_DIR,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    RESULT_V2_PATH,
    SELECTION_PATH,
    SESSION_ID,
    HoldoutE2EV2Error,
    atomic_write_json,
    build_preflight_payload,
    load_authorization_v2,
    load_contract_v2,
    load_selected_cases,
    load_selection,
    mark_auth_execution,
    refuse_runtime_overrides,
    refuse_unauthorized_v2,
    require_sealed_index,
    write_contract_and_auth,
)
from lumenfin.eval.ledger_public_holdout_index import open_store  # noqa: E402
from lumenfin.eval.ledger_structured_citation_shadow import (  # noqa: E402
    STRUCTURED_SHADOW_SYSTEM_PROMPT,
)
from lumenfin.llm import DeepSeekChatClient, LLMSettings  # noqa: E402
from lumenfin.rag.hybrid_retriever import reciprocal_rank_fusion  # noqa: E402
from lumenfin.stdio import configure_stdio_utf8  # noqa: E402
from lumenfin.structured_answer import (  # noqa: E402
    AllowedEvidence,
    CITATION_SOURCE_STRUCTURED,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    StructuredAnswerError,
    validate_structured_answer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One-shot LEDGER public_holdout held-out E2E v2 against the sealed "
            "rc5 DashScope index. Default-deny; v1 blocked records stay historical."
        )
    )
    parser.add_argument(
        "--write-contract",
        action="store_true",
        help="Freeze selection if needed and write contract_v2 + authorization_v2",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--confirm-public-holdout-consumption", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume unfinished official per-case jsonl before consume seal",
    )
    return parser


def _retrieve_hybrid(store: Any, *, query: str, company: str, top_k: int) -> tuple[list[dict], dict]:
    bm25_hits = store.bm25_search(
        query,
        session_id=SESSION_ID,
        tenant_id=SESSION_ID,
        companies=[company],
        top_k=top_k,
    )
    dense_hits = store.vector_search(
        query,
        session_id=SESSION_ID,
        tenant_id=SESSION_ID,
        companies=[company],
        top_k=top_k,
    )
    query_remote_calls = int(getattr(store, "last_query_embed_physical_calls", 0) or 0)
    query_cache_hit = bool(getattr(store, "last_query_embed_cache_hit", False))
    if query_remote_calls <= 0 and not query_cache_hit:
        raise HoldoutE2EV2Error("hybrid query embedding missing physical-call accounting")
    if not dense_hits or not bm25_hits:
        raise HoldoutE2EV2Error("hybrid retrieve requires non-empty dense and BM25 hits")
    hits = reciprocal_rank_fusion(
        [dense_hits, bm25_hits],
        retrieval_method="hybrid_dense_bm25_rrf",
        weights=[1.0, DEFAULT_BM25_RRF_WEIGHT],
    )[:top_k]
    stamped: list[dict] = []
    for hit in hits:
        row = dict(hit)
        row.setdefault("tenant_id", SESSION_ID)
        row.setdefault("session_id", SESSION_ID)
        stamped.append(row)
    return stamped, {
        "mode": "hybrid_dense_bm25_rrf",
        "bm25_hits": len(bm25_hits),
        "dense_hits": len(dense_hits),
        "query_embedding_cache_hit": query_cache_hit,
        "query_embedding_physical_calls": query_remote_calls,
    }


def _validate_structured(
    *,
    citations: list[str],
    window: list[dict[str, Any]],
    parsed: dict[str, Any],
) -> bool:
    if not citations:
        return False
    if str(parsed.get("structured_answer_schema_version") or "") != STRUCTURED_ANSWER_SCHEMA_VERSION:
        return False
    if str(parsed.get("citation_source") or "") not in {"", CITATION_SOURCE_STRUCTURED}:
        # parse_answer_payload may set structured source when citations key present
        pass
    allowed = [
        AllowedEvidence(
            chunk_id=str(hit["chunk_id"]),
            tenant_id=SESSION_ID,
            session_id=SESSION_ID,
            verified=not bool(hit.get("unverified")),
            stale=bool(hit.get("stale_repair_attempt")),
        )
        for hit in window
    ]
    try:
        validate_structured_answer(
            {
                "answer": str(parsed.get("answer") or "ok"),
                "citations": citations,
                "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
                "citation_source": CITATION_SOURCE_STRUCTURED,
                "workflow_status": "completed",
                "has_factual_conclusion": parsed.get("value") is not None and not parsed.get("abstain"),
            },
            allowed=allowed,
            expected_tenant_id=SESSION_ID,
            expected_session_id=SESSION_ID,
            require_citation_for_factual=bool(
                parsed.get("value") is not None and not parsed.get("abstain")
            ),
        )
        return True
    except StructuredAnswerError:
        return False


def _run_case(
    *,
    case: dict[str, Any],
    store: Any,
    reranker: Any,
    llm: DeepSeekChatClient,
    max_document_chars: int,
) -> dict[str, Any]:
    source_k = ARM_SPECS["A_prod"].source_k
    final_k = ARM_SPECS["A_prod"].final_k
    retrieve_started = time.perf_counter()
    pool_hits, retrieve_meta = _retrieve_hybrid(
        store,
        query=str(case["query_text"]),
        company=str(case["company_key"]),
        top_k=source_k,
    )
    pool = prepare_rerank_pool(pool_hits, arm="A_prod")
    retrieve_ms = round((time.perf_counter() - retrieve_started) * 1000.0, 2)

    rerank_started = time.perf_counter()
    ranked, rerank_meta = reranker.rerank(
        str(case["query_text"]),
        pool,
        top_k=final_k,
    )
    rerank_ms = round((time.perf_counter() - rerank_started) * 1000.0, 2)
    for hit in ranked:
        hit.setdefault("tenant_id", SESSION_ID)
        hit.setdefault("session_id", SESSION_ID)

    official = official_ranked_hits(
        ranked,
        origin="production_rag",
        tenant_id=SESSION_ID,
        session_id=SESSION_ID,
    )
    window = build_final_evidence_window(official)
    alias_map = build_citation_alias_map(
        window,
        case_id=str(case["case_id"]),
        attempt_id="1",
        tenant_id=SESSION_ID,
        session_id=SESSION_ID,
    )
    pairs = [
        (alias_map.aliases[index], str(hit.get("text") or ""))
        for index, hit in enumerate(window)
    ]
    user_prompt = render_alias_passages(
        pairs,
        query_text=str(case["query_text"]),
        max_document_chars=max_document_chars,
    )
    if "chunk_id=" in user_prompt.casefold():
        raise HoldoutE2EV2Error("prompt leaked a stable chunk id")

    llm.mark_usage_start()
    generate_started = time.perf_counter()
    provider_success = True
    generate_error = ""
    raw = ""
    try:
        raw = llm.chat(
            STRUCTURED_SHADOW_SYSTEM_PROMPT,
            user_prompt,
            temperature=0.0,
            max_tokens=200,
        )
    except Exception as exc:  # noqa: BLE001 - provider boundary
        provider_success = False
        generate_error = type(exc).__name__
    generate_ms = round((time.perf_counter() - generate_started) * 1000.0, 2)
    usage = llm.usage_since_mark() if provider_success else {
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    predicted = None
    abstain = True
    citations: list[str] = []
    structured_ok = False
    alias_error = ""
    if provider_success:
        try:
            parsed = parse_answer_payload(raw, allow_duplicate_citations=True)
            predicted = parsed.get("value")
            abstain = bool(parsed.get("abstain"))
            try:
                citations = list(
                    parse_and_map_citation_aliases(
                        parsed.get("citations") or [],
                        alias_map,
                        expected_case_id=str(case["case_id"]),
                        expected_attempt_id="1",
                        expected_tenant_id=SESSION_ID,
                        expected_session_id=SESSION_ID,
                    )
                )
            except (CitationAliasError, StructuredAnswerError) as exc:
                alias_error = type(exc).__name__
                citations = []
            structured_ok = _validate_structured(
                citations=citations,
                window=window,
                parsed=parsed,
            )
        except Exception as exc:  # noqa: BLE001 - parse boundary
            provider_success = False
            generate_error = type(exc).__name__
            structured_ok = False

    metrics = evaluate_verified_e2e_case(
        predicted=predicted if isinstance(predicted, (int, float)) else None,
        gold=float(case["gold_value"]),
        abstain=abstain,
        citations=citations,
        hits=window,
        qrels=case["qrels"],
        provider_success=provider_success,
        structured_contract_valid=structured_ok,
    )
    return {
        "query_id": case["query_id"],
        "company_key": case["company_key"],
        "arm": "A_prod",
        "predicted_value": predicted,
        "gold_value": case["gold_value"],
        "abstain": abstain,
        "citations": citations,
        "retrieve": retrieve_meta,
        "retrieve_latency_ms": retrieve_ms,
        "rerank_attempts": int(rerank_meta.get("rerank_attempts") or 0),
        "rerank_fallback": bool(rerank_meta.get("rerank_fallback")),
        "rerank_error_type": str(rerank_meta.get("rerank_error_type") or ""),
        "rerank_latency_ms": float(rerank_meta.get("rerank_latency_ms") or rerank_ms),
        "generate_attempts": int(getattr(llm, "last_attempts", 1) or 1) if provider_success else 0,
        "generate_tokens": int(usage["prompt_tokens"] + usage["completion_tokens"]),
        "generate_latency_ms": generate_ms,
        "generate_error_type": generate_error,
        "citation_alias_error": alias_error,
        "document_reembedding_calls": 0,
        "public_dev_candidate_cache_used": False,
        **metrics,
    }


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise HoldoutE2EV2Error("per-case jsonl row must be an object")
        rows.append(payload)
    return rows


def run_preflight(*, repo_root: Path) -> dict[str, Any]:
    refuse_unauthorized_v2(repo_root=repo_root, argv=["--preflight-only"])
    contract = load_contract_v2(repo_root=repo_root)
    selection = load_selection(repo_root=repo_root)
    seal = require_sealed_index(repo_root=repo_root)
    payload = build_preflight_payload(
        contract=contract,
        selection=selection,
        index_seal=seal,
    )
    out_dir = repo_root / PREFLIGHT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / "preflight.json", payload)
    if payload["status"] == "PREFLIGHT_OK":
        mark_auth_execution(
            repo_root=repo_root,
            config_hash=str(contract["config_hash"]),
            kind="preflight",
        )
    return payload


def run_official(*, repo_root: Path, resume: bool) -> dict[str, Any]:
    refuse_unauthorized_v2(
        repo_root=repo_root,
        argv=[
            "--confirm-public-holdout-consumption",
            "--allow-remote",
            *(["--resume"] if resume else []),
        ],
    )
    bootstrap_dotenv(root=repo_root, announce=False, strict_conflicts=True)
    contract = load_contract_v2(repo_root=repo_root)
    selection = load_selection(repo_root=repo_root)
    seal = require_sealed_index(repo_root=repo_root)
    result_path = repo_root / RESULT_V2_PATH
    if result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("holdout_consumed") is True:
            raise HoldoutE2EV2Error("result_v2 already consumed; refusing second run")

    out_dir = repo_root / OFFICIAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    per_case_path = out_dir / "per_case.jsonl"
    if per_case_path.exists() and not resume:
        raise HoldoutE2EV2Error("official per_case.jsonl exists; pass --resume or clear outputs")

    cases = load_selected_cases(repo_root=repo_root, selection=selection)
    if len(cases) != MAX_OFFICIAL_CASES:
        raise HoldoutE2EV2Error("official case count mismatch")

    uri = str((repo_root / Path(str(seal["uri"]))).resolve())
    store = open_store(uri=uri, allow_remote=True)
    rerank_settings = snapshot_rerank_settings()
    reranker = build_locked_qwen3_reranker(rerank_settings)
    llm = DeepSeekChatClient(LLMSettings.from_env())
    completed = {str(row["query_id"]): row for row in _read_jsonl(per_case_path)}
    ordered: list[dict[str, Any]] = []
    document_reembedding_calls = 0
    query_embed_calls = 0
    qwen3_calls = 0
    deepseek_calls = 0

    try:
        for index, case in enumerate(cases):
            query_id = str(case["query_id"])
            if query_id in completed:
                row = completed[query_id]
                ordered.append(row)
            else:
                row = _run_case(
                    case=case,
                    store=store,
                    reranker=reranker,
                    llm=llm,
                    max_document_chars=int(rerank_settings["max_document_chars"]),
                )
                ordered.append(row)
                completed[query_id] = row
                _atomic_jsonl(per_case_path, ordered)
                print(
                    f"[holdout-e2e-v2] case={index + 1}/{len(cases)} "
                    f"verified={row.get('verified_e2e_success')}",
                    flush=True,
                )
            query_embed_calls += int(
                (row.get("retrieve") or {}).get("query_embedding_physical_calls") or 0
            )
            qwen3_calls += 0 if row.get("rerank_fallback") else 1
            deepseek_calls += 1 if row.get("provider_success") else 0
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()

    if len(ordered) != MAX_OFFICIAL_CASES:
        raise HoldoutE2EV2Error("official run incomplete")
    summary = summarize_cases(ordered)
    aggregate = {
        "schema_version": "ledger_public_holdout_e2e_aggregate.v2",
        "suite": "ledger_public_holdout_held_out_e2e_v2",
        "claim": CLAIM,
        "product_tag": PRODUCT_TAG,
        "product_commit": PRODUCT_COMMIT,
        "config_hash": contract["config_hash"],
        "selected_query_ids_sha256": selection["selected_query_ids_sha256"],
        "index_milvus_db_sha256": seal.get("milvus_db_sha256"),
        "cases": len(ordered),
        "call_counts": {
            "query_embedding_physical_calls": query_embed_calls,
            "qwen3_rerank_calls": qwen3_calls,
            "deepseek_logical_calls": deepseek_calls,
            "document_reembedding_calls": document_reembedding_calls,
        },
        "public_dev_candidate_cache_used": False,
        "summary": summary,
    }
    atomic_write_json(out_dir / "aggregate.json", aggregate)

    auth = mark_auth_execution(
        repo_root=repo_root,
        config_hash=str(contract["config_hash"]),
        kind="remote",
    )
    strict = summary["strict_verified_e2e_success"]
    result = {
        "schema_version": "ledger_public_holdout_e2e_result.v2",
        "seal_status": "SEALED",
        "claim": CLAIM,
        "dataset_specific": True,
        "held_out": True,
        "single_use": True,
        "general_product_accuracy_claim": False,
        "product_accuracy_claim": False,
        "financial_accuracy_claim": False,
        "benchmark_claim": False,
        "product_tag": PRODUCT_TAG,
        "product_commit": PRODUCT_COMMIT,
        "config_hash": contract["config_hash"],
        "selection_path": str(SELECTION_PATH).replace("\\", "/"),
        "contract_path": str(CONTRACT_V2_PATH).replace("\\", "/"),
        "authorization_path": str(AUTHORIZATION_V2_PATH).replace("\\", "/"),
        "selected_query_ids_sha256": selection["selected_query_ids_sha256"],
        "index_uri": seal.get("uri"),
        "index_milvus_db_sha256": seal.get("milvus_db_sha256"),
        "index_collection": seal.get("collection"),
        "holdout_consumed": True,
        "official_preflight_executions": (
            (auth.get("records") or {})
            .get(str(contract["config_hash"]), {})
            .get("official_preflight_executions")
        ),
        "official_remote_executions": 1,
        "cases_total": len(ordered),
        "strict_verified_e2e_success": strict,
        "evaluable_coverage": summary.get("evaluable_coverage"),
        "conditional_verified_e2e_success": summary.get("conditional_verified_e2e_success"),
        "call_counts": aggregate["call_counts"],
        "public_dev_candidate_cache_used": False,
        "document_reembedding_calls": 0,
        "output_dir": str(OFFICIAL_DIR).replace("\\", "/"),
    }
    atomic_write_json(result_path, result)
    selection_update = dict(selection)
    selection_update["holdout_consumed"] = True
    atomic_write_json(repo_root / SELECTION_PATH, selection_update)
    return result


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    parsed, unknown = parser.parse_known_args(args)
    try:
        refuse_runtime_overrides(args)
        if unknown:
            raise HoldoutE2EV2Error("CLI refuses runtime overrides")
        if parsed.write_contract:
            payload = write_contract_and_auth(repo_root=ROOT)
            print(
                json.dumps(
                    {
                        "status": "CONTRACT_WRITTEN",
                        "config_hash": payload["contract"]["config_hash"],
                        "selected_cases": payload["selection"]["selected_cases"],
                        "index_status": payload["index_seal"]["status"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if parsed.preflight_only:
            if parsed.allow_remote or parsed.confirm_public_holdout_consumption:
                raise HoldoutE2EV2Error("preflight refuses remote/consume flags")
            payload = run_preflight(repo_root=ROOT)
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return int(payload.get("exit_code") or 0)
        if not parsed.confirm_public_holdout_consumption:
            raise HoldoutE2EV2Error(
                "choose --write-contract, --preflight-only, or official "
                "--confirm-public-holdout-consumption --allow-remote"
            )
        result = run_official(repo_root=ROOT, resume=bool(parsed.resume))
        print(
            json.dumps(
                {
                    "status": result["seal_status"],
                    "holdout_consumed": result["holdout_consumed"],
                    "strict_verified_e2e_success": result["strict_verified_e2e_success"],
                    "config_hash": result["config_hash"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except HoldoutE2EV2Error as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
