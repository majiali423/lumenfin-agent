"""Fair B1 (retrieve-then-generate) vs B2 (current Agent) runners. Neither reads gold."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from lumenfin.config import AppConfig
from lumenfin.document_ingest import parse_upload_documents
from lumenfin.eval.budget import (
    AUTH_TOKEN,
    EVAL_BUDGET,
    BudgetedLLMClient,
    budget_packet,
    fair_rerun_budget_report,
    get_eval_budget,
)
from lumenfin.eval.document_tasks import ROOT
from lumenfin.eval.gold_evaluator import score_task
from lumenfin.eval.offline_guard import (
    apply_eval_offline_process_env,
    install_outbound_block,
    uninstall_outbound_block,
)
from lumenfin.eval.pricing import estimate_cost_usd
from lumenfin.finrun import export_finrun_state
from lumenfin.graph import LumenFinAgentSystem
from lumenfin.llm import LocalFallbackLLMClient, build_llm_client
from lumenfin.service import LumenFinAnalysisService
from lumenfin.tools import extract_companies_from_query
from lumenfin.tracing import analysis_trace, attach_llm_tracing, completed_analysis_trace, current_trace_events, eval_case_context, patch_remote_run

# Isolated eval index. Not Compose production (DashScope 1024-d + Qwen3).
RETRIEVAL_PROFILE_ID = "lexical_deterministic_eval"
RETRIEVAL_PROFILE = {
    "id": RETRIEVAL_PROFILE_ID,
    "production_equivalent": False,
    "embedding_provider": "deterministic",
    "embedding_dimension": 384,
    "rerank_provider": "lexical",
    "bm25_enabled": False,
    "label": (
        "B1/B2 share the same uploaded documents, isolated Milvus Lite index, "
        "deterministic embeddings, and lexical rerank. This is a retrieval-config "
        "contrast, not the approved local production profile (DashScope "
        "text-embedding-v4 1024-d + qwen3 rerank + versioned BM25 collection)."
    ),
}

B1_SYSTEM_PROMPT = (
    "You answer using only the retrieved passages. Cite filenames and page numbers. "
    "If the passages are insufficient, say so and do not invent numbers. "
    "Do not use outside knowledge or sample databases."
)


def repo_identity(repo_root: Path) -> dict[str, Any]:
    def _git(*args: str) -> str:
        proc = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout or "").strip()

    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def _eval_config(root: Path) -> AppConfig:
    db_path = root / "data" / "lumenfin.db"
    base = AppConfig.from_env()
    return replace(
        base,
        output_dir=root / "outputs",
        upload_dir=root / "uploads",
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        redis_url=None,
        api_key=None,
        app_env="test",
        data_mode="live",
        allow_local_fallback=True,
        rag_enabled=True,
        rag_index_mode="sync_on_run",
        fetch_live_fundamentals=False,
        fetch_sec_fundamentals=False,
        milvus_uri=str(root / "milvus_eval.db"),
        milvus_isolate=True,
        embedding_provider=RETRIEVAL_PROFILE["embedding_provider"],
        embedding_dimension=int(RETRIEVAL_PROFILE["embedding_dimension"]),
        rag_rerank_provider=RETRIEVAL_PROFILE["rerank_provider"],
        rag_rerank_base_url="",
        rag_bm25_enabled=bool(RETRIEVAL_PROFILE["bm25_enabled"]),
        bounded_repair_enabled=False,
        profile_llm_max_attempts=1,
    )


def _llm(*, offline: bool):
    if offline:
        apply_eval_offline_process_env()
        return LocalFallbackLLMClient()
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("live eval requires DEEPSEEK_API_KEY in process env")
    from dataclasses import replace as dc_replace

    settings = dc_replace(
        AppConfig.from_env().llm,
        model=str(EVAL_BUDGET["model_id"]),
        max_retries=int(EVAL_BUDGET["max_attempts_per_call"]),
        timeout_seconds=float(EVAL_BUDGET["timeout_seconds"]),
    )
    client = build_llm_client(settings, allow_local_fallback=False)
    if str(getattr(client, "backend_name", "")) == "local-fallback":
        raise RuntimeError("live eval refused silent local fallback")
    budget = get_eval_budget()
    if budget is None:
        raise RuntimeError("live eval requires an EvalBudget on the context")
    return BudgetedLLMClient(client, budget, budget.spec)


def _backend_name(llm: Any) -> str:
    return str(getattr(llm, "backend_name", getattr(llm, "model_name", "unknown")))


def result_class(*, offline: bool, backend: str) -> dict[str, Any]:
    fallback = offline or backend == "local-fallback"
    return {
        "accuracy_eligible": False,
        "result_class": "offline_local_fallback" if fallback else "live_dev_diagnostic",
        "llm_backend": backend,
        "retrieval_profile": RETRIEVAL_PROFILE_ID,
        "retrieval_production_equivalent": False,
        "scope": (
            "excerpt_level_dev_contrast; not full 10-K eval; not production "
            "DashScope/Qwen3 retrieval; B1/B2 share documents and initial retrieval"
        ),
        "note": (
            "Development diagnostic under candidate gold. Not formal accuracy."
            if not fallback
            else "Offline LocalFallback is execution/scorer coverage only."
        ),
    }


def run_input_from_task(task: dict[str, Any]) -> dict[str, Any]:
    """Product-side fields only. Gold labels are not inputs."""
    return {
        "id": task.get("id"),
        "query": task.get("query"),
        "allowed_documents": list(task.get("allowed_documents") or []),
        "user_scope": dict(task.get("user_scope") or {}),
    }


def companies_from_run_input(run_input: dict[str, Any]) -> list[str]:
    query = str(run_input.get("query") or "")
    extra = list((run_input.get("user_scope") or {}).get("companies") or [])
    found = extract_companies_from_query(query, document_contexts=None, llm_client=None)
    for name in extra:
        if name and name not in found:
            found.append(str(name))
    return found


def b1_retrieval_plan(run_input: dict[str, Any]) -> dict[str, Any]:
    companies = companies_from_run_input(run_input)
    mode = "query_companies" if companies else "all_uploaded_docs"
    return {
        "query": run_input.get("query"),
        "document_ids": [str(doc.get("document_id") or doc.get("path")) for doc in run_input.get("allowed_documents") or []],
        "companies": companies,
        "mode": mode,
    }


def _load_docs(run_input: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    paths: list[str] = []
    contexts: list[dict[str, Any]] = []
    for doc in run_input.get("allowed_documents") or []:
        path = ROOT / doc["path"]
        paths.append(str(path))
        contexts.extend(parse_upload_documents(path))
    return paths, contexts


def run_b1(task: dict[str, Any], *, root: Path, offline: bool) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ.setdefault("MAS_TRACE_DIR", str(root / "traces"))
    if offline:
        apply_eval_offline_process_env()
        install_outbound_block()
    try:
        return _run_b1_body(task, root=root, offline=offline, started=started)
    finally:
        if offline:
            uninstall_outbound_block()


def _run_b1_body(task: dict[str, Any], *, root: Path, offline: bool, started: float) -> dict[str, Any]:
    llm = attach_llm_tracing(_llm(offline=offline))
    config = _eval_config(root)
    system = LumenFinAgentSystem(llm_client=llm, app_config=config)
    run_input = run_input_from_task(task)
    paths, contexts = _load_docs(run_input)
    plan = b1_retrieval_plan(run_input)
    hits: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"mode": plan["mode"], "companies": plan["companies"]}
    if plan["companies"]:
        merged: dict[str, dict[str, Any]] = {}
        for company in plan["companies"]:
            company_hits, company_meta = system.runtime.hybrid_retriever.retrieve_for_company_with_meta(
                query=str(run_input["query"]),
                company=company,
                session_id=f"b1-{run_input['id']}",
                document_contexts=contexts,
            )
            meta[f"meta_{company}"] = company_meta
            for hit in company_hits:
                key = str(hit.get("chunk_id") or id(hit))
                prev = merged.get(key)
                if prev is None or float(hit.get("score") or 0) >= float(prev.get("score") or 0):
                    merged[key] = hit
        hits = sorted(merged.values(), key=lambda item: float(item.get("score") or 0), reverse=True)
    else:
        hits, meta = system.runtime.hybrid_retriever.retrieve_for_company_with_meta(
            query=str(run_input["query"]),
            company="",
            session_id=f"b1-{run_input['id']}",
            document_contexts=contexts,
        )
        meta["mode"] = "all_uploaded_docs"
    passages = []
    for hit in hits[: config.rag_top_k]:
        passages.append(
            f"[{hit.get('filename') or 'doc'}#p{hit.get('page') or 1}] {_clip(hit.get('text') or hit.get('chunk') or '')}"
        )
    user = task["query"] + "\n\nPassages:\n" + "\n".join(passages)
    with analysis_trace(
        name="lumenfin.eval.b1",
        metadata={
            "task_id": task["id"],
            "system": "b1_rag",
            "model": getattr(llm, "model_name", "unknown"),
        },
        inputs={"query": task["query"], "task_id": task["id"]},
    ) as span:
        output = llm.chat(B1_SYSTEM_PROMPT, user, temperature=0.0, max_tokens=700)
        span["outputs"] = {
            "final_output_preview": _clip(output, 400),
            "usage": getattr(llm, "_usage_totals", {}),
        }
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    usage = llm.usage_since_mark() if hasattr(llm, "usage_since_mark") else {}
    llm.mark_usage_start()
    usage = {
        "prompt_tokens": getattr(llm, "_usage_totals", {}).get("prompt_tokens"),
        "completion_tokens": getattr(llm, "_usage_totals", {}).get("completion_tokens"),
    }
    trace = completed_analysis_trace()
    return {
        "system": "b1_rag",
        "task_id": task["id"],
        "final_output": output,
        "workflow_status": "completed",
        "retrieval": {
            "mode": meta.get("mode"),
            "hit_count": len(hits),
            "chunk_ids": [h.get("chunk_id") for h in hits[:10]],
            "companies": plan["companies"],
        },
        "duration_ms": duration_ms,
        "usage": usage,
        "cost": estimate_cost_usd(
            model=str(getattr(llm, "model_name", "unknown")),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        ),
        "document_paths": paths,
        "error": None,
        "retrieval_profile": RETRIEVAL_PROFILE,
        "langsmith_url": span.get("langsmith_url") or trace.get("langsmith_url"),
        "langsmith_run_id": span.get("langsmith_run_id") or trace.get("langsmith_run_id"),
        "langsmith_remote": span.get("remote") or trace.get("remote"),
        "local_trace_id": span.get("run_id") or trace.get("run_id"),
        **result_class(offline=offline, backend=_backend_name(llm)),
    }


def run_b2(task: dict[str, Any], *, root: Path, offline: bool) -> dict[str, Any]:
    started = time.perf_counter()
    os.environ.setdefault("MAS_TRACE_DIR", str(root / "traces"))
    if offline:
        apply_eval_offline_process_env()
        install_outbound_block()
    try:
        return _run_b2_body(task, root=root, offline=offline, started=started)
    finally:
        if offline:
            uninstall_outbound_block()


def _run_b2_body(task: dict[str, Any], *, root: Path, offline: bool, started: float) -> dict[str, Any]:
    llm = attach_llm_tracing(_llm(offline=offline))
    config = _eval_config(root)
    service = LumenFinAnalysisService(config, llm_client=llm)
    run_input = run_input_from_task(task)
    paths, _contexts = _load_docs(run_input)
    with eval_case_context(
        {
            "task_id": run_input["id"],
            "system": "b2_agent",
            "model": getattr(llm, "model_name", "unknown"),
        }
    ):
        payload = service.analyze(
            query=str(run_input["query"]),
            thread_id=f"b2-{run_input['id']}-{uuid4().hex[:8]}",
            export_artifacts=False,
            document_paths=paths,
        )
    trace = completed_analysis_trace()
    state = payload.get("result") or {}
    duration_ms = round((time.perf_counter() - started) * 1000, 3)
    output = str(state.get("final_report") or state.get("final_output") or "")
    usage = _collect_usage(llm, getattr(service, "_last_run_llm", None))
    finrun = None
    try:
        finrun = export_finrun_state(state)
    except Exception:
        finrun = None
    telemetry = state.get("run_telemetry") or {}
    return {
        "system": "b2_agent",
        "task_id": task["id"],
        "final_output": output,
        "workflow_status": state.get("workflow_status") or payload.get("workflow_status") or "completed",
        "missing_fields": state.get("missing_fields"),
        "clarification_questions": state.get("clarification_questions"),
        "internal_metrics": (state.get("computed_metrics") or {}).get("rows")
        if isinstance(state.get("computed_metrics"), dict)
        else state.get("computed_metrics"),
        "finrun": finrun,
        "duration_ms": duration_ms,
        "usage": usage,
        "cost": estimate_cost_usd(
            model=str(getattr(llm, "model_name", "unknown")),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        ),
        "retrieval": {
            "mode": (telemetry.get("rag") or telemetry).get("mode")
            if isinstance(telemetry, dict)
            else None
        },
        "document_paths": paths,
        "error": None,
        "trace_events": trace.get("events") or current_trace_events(),
        "retrieval_profile": RETRIEVAL_PROFILE,
        "langsmith_url": trace.get("langsmith_url"),
        "langsmith_run_id": trace.get("langsmith_run_id"),
        "langsmith_remote": trace.get("remote"),
        "local_trace_id": trace.get("run_id"),
        **result_class(offline=offline, backend=_backend_name(llm)),
    }


def _clip(text: str, limit: int = 900) -> str:
    value = " ".join(str(text).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def _collect_usage(parent: Any, forked: Any | None) -> dict[str, Any]:
    sources = [item for item in (forked, parent) if item is not None]
    prompt = 0
    completion = 0
    seen = False
    for client in sources:
        totals = getattr(client, "_usage_totals", None)
        if not isinstance(totals, dict):
            continue
        p = int(totals.get("prompt_tokens") or 0)
        c = int(totals.get("completion_tokens") or 0)
        if p or c:
            seen = True
            prompt = max(prompt, p)
            completion = max(completion, c)
    if seen:
        return {"prompt_tokens": prompt, "completion_tokens": completion, "status": "recorded"}
    event_prompt = 0
    event_completion = 0
    for event in current_trace_events():
        if event.get("name") != "llm.chat":
            continue
        if event.get("prompt_tokens") or event.get("completion_tokens"):
            event_prompt += int(event.get("prompt_tokens") or 0)
            event_completion += int(event.get("completion_tokens") or 0)
    if event_prompt or event_completion:
        return {
            "prompt_tokens": event_prompt,
            "completion_tokens": event_completion,
            "status": "recorded_from_trace_events",
        }
    return {"prompt_tokens": None, "completion_tokens": None, "status": "unknown"}
    value = " ".join(str(text).split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def score_run(task: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    metrics = run.get("internal_metrics")
    rows: list[dict[str, Any]] = []
    if isinstance(metrics, dict):
        for company, payload in metrics.items():
            if not isinstance(payload, dict):
                continue
            market = payload.get("market_data") or payload
            if isinstance(market, dict):
                for name, value in market.items():
                    if isinstance(value, (int, float)):
                        rows.append({"entity": company, "name": name, "value": value})
    elif isinstance(metrics, list):
        rows = [item for item in metrics if isinstance(item, dict)]
    scored = score_task(
        task,
        final_output=str(run.get("final_output") or ""),
        workflow_status=str(run.get("workflow_status") or "completed"),
        citations=list((run.get("finrun") or {}).get("citations") or [])
        if isinstance(run.get("finrun"), dict)
        else None,
        internal_metrics=rows or None,
        finrun=run.get("finrun") if isinstance(run.get("finrun"), dict) else None,
        error=run.get("error"),
        system=str(run.get("system") or ""),
    )
    merged = {**run, "gold": scored}
    try:
        patch_remote_run(
            str(run.get("langsmith_run_id") or "") or None,
            outputs={
                "gold_passed": scored.get("passed"),
                "formal_pass": scored.get("formal_pass"),
                "expected_action": scored.get("expected_action"),
                "findings": scored.get("findings"),
                "usage": run.get("usage"),
                "task_id": run.get("task_id"),
                "system": run.get("system"),
            },
            extra_metadata={
                "task_id": run.get("task_id"),
                "system": run.get("system"),
                "gold_passed": scored.get("passed"),
            },
        )
    except Exception:
        pass
    return merged


def planned_live_budget(n_tasks: int = 24) -> dict[str, Any]:
    return budget_packet(n_tasks=n_tasks, smoke=n_tasks == 2)


def live_experiment_packet() -> dict[str, Any]:
    from lumenfin.env_bootstrap import describe_credential_sources

    keys = describe_credential_sources(
        root=ROOT,
        keys=("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY", "LANGSMITH_API_KEY"),
    )
    budget = budget_packet(n_tasks=24)
    deepseek = next(item for item in keys if item.key == "DEEPSEEK_API_KEY")
    langsmith = next(item for item in keys if item.key == "LANGSMITH_API_KEY")
    auth = (os.getenv("LUMENFIN_EVAL_BUDGET_AUTH") or "").strip()
    return {
        "authorization_request": (
            f"Set LUMENFIN_EVAL_BUDGET_AUTH={AUTH_TOKEN} in this session. That token "
            f"authorizes a cumulative cap of {budget['max_provider_requests']} DeepSeek "
            "HTTP requests for the 24-task lexical excerpt pilot, not a fresh quota "
            "each time --out changes. --confirm-budget is not authorization."
        ),
        "entry_smoke": (
            "python scripts/run_document_task_eval.py --pilot --allow-live --smoke "
            "--out outputs/document_task_eval/pilot_live_smoke.json"
        ),
        "entry_remainder": (
            "python scripts/run_document_task_eval.py --pilot --allow-live "
            "--out outputs/document_task_eval/pilot_live.json"
        ),
        "systems": ["b1_rag", "b2_agent"],
        "tasks": 24,
        "runs": 48,
        "model": {
            "provider": "deepseek",
            "id": EVAL_BUDGET["model_id"],
            "aliases_accepted": EVAL_BUDGET["model_aliases_accepted"],
            "served_as": budget["pricing"]["served_as"],
            "allow_local_fallback": False,
        },
        "retrieval": RETRIEVAL_PROFILE,
        "scope": {
            "issuers": ["NVIDIA", "Microsoft", "Apple"],
            "documents": "committed derived excerpts only",
            "full_10k": False,
            "production_retrieval": False,
            "b1_b2_same_initial_retrieval": True,
        },
        "budget": budget,
        "accumulated": fair_rerun_budget_report(ROOT / "outputs" / "document_task_eval" / "budget_state.json"),
        "credentials": [
            {"key": item.key, "source": item.source, "length": item.length} for item in keys
        ],
        "langsmith": (
            "optional_local_only" if langsmith.length == 0 else "key_present_probe_required"
        ),
        "budget_state": "outputs/document_task_eval/budget_state.json",
        "remaining_conditions": _live_conditions(deepseek.length > 0, auth_ok=auth == AUTH_TOKEN),
    }


def _live_conditions(deepseek_present: bool, *, auth_ok: bool) -> list[str]:
    missing: list[str] = []
    if not deepseek_present:
        missing.append("DEEPSEEK_API_KEY present (value not displayed)")
    if not auth_ok:
        missing.append(
            f"LUMENFIN_EVAL_BUDGET_AUTH={AUTH_TOKEN} (user-approved budget; not --confirm-budget)"
        )
    return missing


def write_ledger(path: Path, rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "lumenfin_document_task_ledger.v1",
        "identity": {
            "lumenfin": repo_identity(ROOT),
            "finagentbench": repo_identity(ROOT.parent / "finagentbench-demo")
            if (ROOT.parent / "finagentbench-demo").exists()
            else None,
        },
        "extra": extra or {},
        "cases": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
