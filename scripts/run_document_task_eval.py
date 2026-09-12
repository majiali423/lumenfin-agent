#!/usr/bin/env python3
"""Document-task eval: independent gold, B1 RAG vs B2 Agent, local ledger.

Does not consume FinanceBench confirmation/holdout. Live runs require
LUMENFIN_EVAL_BUDGET_AUTH matching the printed token. --confirm-budget is not
authorization. Offline LocalFallback is not accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("APP_ENV", "test")

_PRE = argparse.ArgumentParser(add_help=False)
_PRE.add_argument("--offline", action="store_true")
_PRE.add_argument("--allow-live", action="store_true")
_EARLY, _ = _PRE.parse_known_args()
if _EARLY.offline:
    from lumenfin.eval.offline_guard import apply_eval_offline_process_env

    apply_eval_offline_process_env()

from lumenfin.eval.baselines import (  # noqa: E402
    live_experiment_packet,
    planned_live_budget,
    run_b1,
    run_b2,
    score_run,
    write_ledger,
)
from lumenfin.eval.budget import (  # noqa: E402
    AUTH_TOKEN,
    EVAL_BUDGET,
    BudgetExhausted,
    EvalBudget,
    SMOKE_TASK_IDS,
    budget_packet,
    fair_rerun_budget_report,
    get_eval_budget,
    load_eval_budget,
    set_eval_budget,
)
from lumenfin.eval.document_tasks import freeze_readiness, load_catalog, tasks_for  # noqa: E402
from lumenfin.tracing import enable_live_eval_tracing, probe_langsmith_remote  # noqa: E402

BUDGET_STATE = ROOT / "outputs" / "document_task_eval" / "budget_state.json"


def _summarize(rows: list[dict], *, offline: bool) -> dict:
    by_family: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "pass": 0, "fail": 0, "error": 0})
    by_system: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "pass": 0, "answerable_n": 0, "answerable_pass": 0, "recorded": 0}
    )
    latencies: dict[str, list[float]] = defaultdict(list)
    scorer_complete = 0
    for row in rows:
        gold = row.get("gold") or {}
        family = str(gold.get("family") or row.get("family") or "unknown")
        system = str(row.get("system") or "unknown")
        by_family[family]["n"] += 1
        if row.get("error"):
            by_family[family]["error"] += 1
        elif gold.get("passed"):
            by_family[family]["pass"] += 1
        else:
            by_family[family]["fail"] += 1
        by_system[system]["n"] += 1
        by_system[system]["recorded"] += 1
        if gold:
            scorer_complete += 1
        if gold.get("passed"):
            by_system[system]["pass"] += 1
        if gold.get("expected_action") == "answer":
            by_system[system]["answerable_n"] += 1
            if gold.get("passed"):
                by_system[system]["answerable_pass"] += 1
        if row.get("duration_ms") is not None:
            latencies[system].append(float(row["duration_ms"]))

    def _pct(hits: int, total: int) -> str:
        if total <= 0:
            return "n/a"
        return f"{hits}/{total}"

    systems = {}
    for name, counts in by_system.items():
        values = sorted(latencies.get(name) or [])
        p50 = values[len(values) // 2] if values else None
        p95 = values[int(max(0, round(0.95 * (len(values) - 1))))] if values else None
        payload = {
            "runs_recorded": _pct(counts["recorded"], counts["n"]),
            "scorer_applied": True,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
        }
        if offline:
            payload["gold_pass_diagnostic_not_accuracy"] = _pct(counts["pass"], counts["n"])
            payload["local_fallback"] = True
        else:
            payload["supported_task_completion"] = _pct(counts["answerable_pass"], counts["answerable_n"])
            payload["all_tasks"] = _pct(counts["pass"], counts["n"])
            payload["denominator"] = (
                "answerable_n is expected_action=answer tasks; refuse/clarify "
                "are reported separately in gold.expected_action"
            )
        systems[name] = payload
    return {
        "by_family": dict(by_family),
        "systems": systems,
        "planned": len(rows),
        "recorded": len(rows),
        "scorer_rows": scorer_complete,
        "accuracy_eligible": False,
        "result_class": "offline_local_fallback" if offline else "live_dev_diagnostic",
        "formal_pass_n": sum(1 for row in rows if (row.get("gold") or {}).get("formal_pass")),
        "diagnostic_pass_n": sum(
            1
            for row in rows
            if (row.get("gold") or {}).get("diagnostic_pass")
            or (row.get("gold") or {}).get("passed")
        ),
        "eval_acceptance_v1_n": sum(
            1 for row in rows if (row.get("gold") or {}).get("eval_acceptance_v1")
        ),
        "eval_acceptance_v1_rule": (
            "Counts gold.task_result.passed with contract_result in "
            "{passed, not_applicable}. Does not change diagnostic_pass_n."
        ),
        "scoring_policy_version": "lumenfin_eval_contract.v1",
        "formal_accuracy_eligible": False,
        "scope": "excerpt_dev_lexical_contrast",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent document-task evaluation")
    parser.add_argument("--pilot", action="store_true", help="24 source-checked dev pilot tasks")
    parser.add_argument("--split", choices=["dev", "test"], default=None)
    parser.add_argument("--systems", default="b1,b2")
    parser.add_argument("--offline", action="store_true", help="LocalFallbackLLM; not a live quality result")
    parser.add_argument("--allow-live", action="store_true")
    parser.add_argument("--confirm-budget", action="store_true", help="Ignored; not authorization")
    parser.add_argument("--smoke", action="store_true", help="Preselected 2 tasks x B1/B2")
    parser.add_argument("--exclude-ids", default="", help="Comma task ids already counted in smoke")
    parser.add_argument("--prepare-live", action="store_true")
    parser.add_argument("--verify-langsmith", action="store_true", help="Create/read a LangSmith run; no DeepSeek")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--budget-only", action="store_true")
    parser.add_argument(
        "--fair-rerun-budget",
        action="store_true",
        help="Print remaining HTTP on the accumulated 960 cap; no model requests",
    )
    parser.add_argument("--freeze-check", action="store_true")
    parser.add_argument("--out", default="outputs/document_task_eval/ledger.json")
    args = parser.parse_args()

    catalog = load_catalog()
    if args.freeze_check:
        payload = freeze_readiness(catalog)
        print(json.dumps(payload, indent=2))
        return int(payload["exit_code"])

    if args.fair_rerun_budget:
        print(json.dumps(fair_rerun_budget_report(BUDGET_STATE), indent=2))
        return 0

    if args.budget_only or args.prepare_live:
        packet = live_experiment_packet() if args.prepare_live else planned_live_budget(24)
        packet["accumulated"] = fair_rerun_budget_report(BUDGET_STATE)
        print(json.dumps(packet, indent=2))
        return 0

    if args.verify_langsmith:
        enable_live_eval_tracing()
        probe = probe_langsmith_remote()
        print(json.dumps(probe, indent=2))
        return 0 if probe.get("ok") else 2

    offline = args.offline or not args.allow_live
    langsmith_probe: dict | None = None
    if args.allow_live:
        if offline:
            print("--allow-live cannot be combined into offline mode", file=sys.stderr)
            return 2
        if not os.getenv("DEEPSEEK_API_KEY", "").strip():
            print("DEEPSEEK_API_KEY missing; refusing --allow-live (value not displayed)", file=sys.stderr)
            return 2
        if args.confirm_budget:
            print("--confirm-budget is not authorization; ignored", file=sys.stderr)
        enable_live_eval_tracing()
        langsmith_probe = probe_langsmith_remote()
        if not langsmith_probe.get("ok"):
            print(json.dumps({"langsmith_probe": langsmith_probe}, indent=2))
            print("refusing live run: LangSmith probe did not create+read a run", file=sys.stderr)
            return 2
        auth = (os.getenv("LUMENFIN_EVAL_BUDGET_AUTH") or "").strip()
        if auth != AUTH_TOKEN:
            print(json.dumps(live_experiment_packet(), indent=2))
            print("refusing live run: missing user budget authorization token", file=sys.stderr)
            return 2
        os.environ["MAS_LLM_MAX_INFLIGHT_PER_PROCESS"] = str(EVAL_BUDGET["max_inflight"])
        budget = load_eval_budget(BUDGET_STATE)
        set_eval_budget(budget)
        print(json.dumps(fair_rerun_budget_report(BUDGET_STATE), indent=2))

    tasks = tasks_for(split=args.split, pilot_only=args.pilot)
    if args.pilot and not args.smoke:
        smoke = [row for row in tasks if row["id"] in SMOKE_TASK_IDS]
        rest = [row for row in tasks if row["id"] not in SMOKE_TASK_IDS]
        tasks = smoke + rest
    if args.smoke:
        tasks = [row for row in tasks if row["id"] in SMOKE_TASK_IDS]
    excluded = {item.strip() for item in args.exclude_ids.split(",") if item.strip()}
    if excluded:
        tasks = [row for row in tasks if row["id"] not in excluded]
    if args.max_cases:
        tasks = tasks[: args.max_cases]
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    runners = {"b1": run_b1, "b2": run_b2}
    rows: list[dict] = []
    out = Path(args.out)
    skip_rest = False
    skip_reason = ""
    planned_pairs = [(task, name) for task in tasks for name in systems]
    for task, name in planned_pairs:
        runner = runners[name]
        work = ROOT / "test_artifacts" / f"dteval-{name}-{task['id']}-{uuid4().hex[:8]}"
        work.mkdir(parents=True, exist_ok=True)
        if skip_rest:
            run = {
                "system": f"{name}_{'rag' if name=='b1' else 'agent'}",
                "task_id": task["id"],
                "final_output": "",
                "error": f"skipped_after_budget:{skip_reason}",
                "workflow_status": "budget_exhausted",
                "accuracy_eligible": False,
                "result_class": "skipped_budget",
            }
        else:
            try:
                run = runner(task, root=work, offline=offline)
                if (not offline) and str(run.get("llm_backend") or "") == "local-fallback":
                    raise RuntimeError("silent local fallback")
            except BudgetExhausted as exc:
                skip_rest = True
                skip_reason = str(exc)
                run = {
                    "system": f"{name}_{'rag' if name=='b1' else 'agent'}",
                    "task_id": task["id"],
                    "final_output": "",
                    "error": f"BudgetExhausted: {exc}",
                    "workflow_status": "budget_exhausted",
                    "accuracy_eligible": False,
                    "result_class": "budget_exhausted",
                }
            except Exception as exc:
                run = {
                    "system": f"{name}_{'rag' if name=='b1' else 'agent'}",
                    "task_id": task["id"],
                    "final_output": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "workflow_status": "error",
                    "accuracy_eligible": False,
                    "result_class": "run_error",
                }
        rows.append(score_run(task, run))
        extra = {
            "offline": offline,
            "catalog_version": catalog["version"],
            "accuracy_eligible": False,
            "result_class": "offline_local_fallback" if offline else "live_dev_diagnostic",
            "retrieval_profile": "lexical_deterministic_eval",
            "retrieval_production_equivalent": False,
            "scope": "excerpt_dev_lexical_contrast",
            "formal_accuracy_eligible": False,
            "scoring_policy_version": "lumenfin_eval_contract.v1",
        }
        budget = get_eval_budget()
        if budget is not None:
            extra["budget"] = budget.snapshot()
            try:
                budget.save(BUDGET_STATE)
            except Exception:
                pass
        write_ledger(out, rows, extra=extra)

    summary = _summarize(rows, offline=offline)
    summary_path = out.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "summary": summary,
                "offline": offline,
                "accuracy_eligible": False,
                "retrieval_profile": "lexical_deterministic_eval",
                "langsmith_probe": langsmith_probe,
                "langsmith_case_urls": [
                    {"task_id": row.get("task_id"), "system": row.get("system"), "url": row.get("langsmith_url")}
                    for row in rows
                    if row.get("langsmith_url")
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
