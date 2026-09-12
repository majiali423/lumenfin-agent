#!/usr/bin/env python3
"""Rescore an existing document-task ledger offline. Does not call paid models."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.baselines import write_ledger
from lumenfin.eval.budget import fair_rerun_budget_report
from lumenfin.eval.document_tasks import load_catalog
from lumenfin.eval.gold_evaluator import score_task
from lumenfin.eval.pricing import estimate_cost_usd

VERSION = "pilot_live_rescore.v2"


def _usage_status(row: dict) -> dict:
    usage = dict(row.get("usage") or {})
    system = str(row.get("system") or "")
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if system == "b2_agent" and (prompt in {0, None} and completion in {0, None}):
        events = row.get("trace_events") or []
        recovered_p = sum(int(item.get("prompt_tokens") or 0) for item in events if item.get("name") == "llm.chat")
        recovered_c = sum(int(item.get("completion_tokens") or 0) for item in events if item.get("name") == "llm.chat")
        if recovered_p or recovered_c:
            return {
                "prompt_tokens": recovered_p,
                "completion_tokens": recovered_c,
                "status": "recovered_from_local_trace_events",
            }
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "status": "unknown",
            "note": "B2 tokens were recorded as 0 because the eval wrapper read the unforked client.",
        }
    if prompt is None or completion is None:
        usage["status"] = usage.get("status") or "unknown"
        return usage
    usage["status"] = usage.get("status") or "recorded"
    return usage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/document_task_eval/pilot_live.json")
    parser.add_argument("--out", default="outputs/document_task_eval/pilot_live_rescore_v2.json")
    args = parser.parse_args()
    source = Path(args.source)
    original = json.loads(source.read_text(encoding="utf-8"))
    catalog = load_catalog()
    tasks = {row["id"]: row for row in catalog["tasks"]}
    changes: list[dict] = []
    rows: list[dict] = []
    by_system = defaultdict(lambda: {"n": 0, "old_pass": 0, "new_pass": 0, "flipped": 0})
    b1_prompt = b1_completion = 0
    b2_known = 0
    b2_unknown = 0
    for case in original.get("cases") or []:
        task = tasks[str(case["task_id"])]
        old = dict(case.get("gold") or {})
        usage = _usage_status(case)
        run = {**case, "usage": usage}
        citations = None
        if isinstance(case.get("finrun"), dict):
            citations = list((case.get("finrun") or {}).get("citations") or [])
        new = score_task(
            task,
            final_output=str(case.get("final_output") or ""),
            workflow_status=str(case.get("workflow_status") or "completed"),
            citations=citations,
            finrun=case.get("finrun") if case.get("system") == "b2_agent" else None,
            error=case.get("error"),
        )
        scored = {**run, "gold": new}
        scored["cost"] = estimate_cost_usd(
            model=str(case.get("cost", {}).get("model") or "deepseek-flash"),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
        new = scored["gold"]
        old_pass = bool(old.get("passed"))
        new_pass = bool(new.get("passed"))
        system = str(case.get("system"))
        by_system[system]["n"] += 1
        by_system[system]["old_pass"] += int(old_pass)
        by_system[system]["new_pass"] += int(new_pass)
        reasons = []
        if old_pass != new_pass:
            by_system[system]["flipped"] += 1
            reasons.append("pass_changed")
        if bool(old.get("formal_pass")) and not new.get("formal_pass"):
            reasons.append("formal_pass_cleared")
        if set(old.get("findings") or []) != set(new.get("findings") or []):
            reasons.append("findings_changed")
        if reasons:
            changes.append(
                {
                    "task_id": case.get("task_id"),
                    "system": system,
                    "old_passed": old_pass,
                    "new_passed": new_pass,
                    "old_formal_pass": old.get("formal_pass"),
                    "new_formal_pass": new.get("formal_pass"),
                    "old_findings": old.get("findings"),
                    "new_findings": new.get("findings"),
                    "reasons": reasons,
                }
            )
        if system == "b1_rag" and usage.get("prompt_tokens") is not None:
            b1_prompt += int(usage.get("prompt_tokens") or 0)
            b1_completion += int(usage.get("completion_tokens") or 0)
        if system == "b2_agent":
            if usage.get("status") == "unknown":
                b2_unknown += 1
            else:
                b2_known += 1
        rows.append(scored)

    summary = {
        "schema_version": VERSION,
        "source_ledger": str(source).replace("\\", "/"),
        "source_not_overwritten": True,
        "accuracy_eligible": False,
        "architecture_fair": False,
        "architecture_fair_reason": (
            "Original B1 runs selected retrieval companies from gold required_facts. "
            "Rescoring cannot remove that contamination."
        ),
        "formal_pass_n": sum(1 for row in rows if (row.get("gold") or {}).get("formal_pass")),
        "diagnostic_pass_n": sum(1 for row in rows if (row.get("gold") or {}).get("diagnostic_pass")),
        "by_system": dict(by_system),
        "changes_n": len(changes),
        "usage": {
            "http_from_original_budget": (original.get("extra") or {}).get("budget"),
            "b1_prompt_tokens": b1_prompt,
            "b1_completion_tokens": b1_completion,
            "b2_token_rows_recovered": b2_known,
            "b2_token_rows_unknown": b2_unknown,
            "round_total_usd": None,
            "round_total_usd_reason": (
                "B2 wrapper tokens were 0. Some local trace events contain partial "
                "llm.chat usage and must not be summed as complete B2 usage. "
                "No whole-run fee is reported."
            ),
            "pricing": {
                "model": "deepseek-flash",
                "source_url": "https://api-docs.deepseek.com/quick_start/pricing",
                "retrieved_at": "2026-09-10",
                "assumption": "cache-miss off-peak; not a hard cap",
            },
        },
        "fair_rerun": {
            **fair_rerun_budget_report(ROOT / "outputs" / "document_task_eval" / "budget_state.json"),
            "note": (
                "Do not treat this rescore as a B1/B2 architecture contrast. "
                "Remaining HTTP is the unused portion of the original 960 cap."
            ),
        },
    }
    extra = {
        **(original.get("extra") or {}),
        "rescore_version": VERSION,
        "accuracy_eligible": False,
        "formal_accuracy_eligible": False,
        "architecture_fair": False,
        "rescore_summary": summary,
        "rescore_changes": changes,
    }
    out = Path(args.out)
    write_ledger(out, rows, extra=extra)
    report = out.with_name(out.stem + "_report.json")
    report.write_text(json.dumps({"summary": summary, "changes": changes}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("report", str(report))
    print("ledger", str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
