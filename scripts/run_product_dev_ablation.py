#!/usr/bin/env python3
"""Offline product-dev ablation on split=dev only. Does not touch public_holdout."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.offline_env import apply_offline_env

apply_offline_env()

from lumenfin import LumenFinAgentSystem
from lumenfin.eval.product_dev import catalog_items, dump_catalog_json, score_item
from lumenfin.llm import LocalFallbackLLMClient
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


MODES = (
    "oneshot",
    "graph_legacy_fatal",
    "graph_taskspec",
    "graph_taskspec_repair",
)


def _oneshot(query: str) -> dict:
    llm = LocalFallbackLLMClient()
    text = llm.chat("Answer the financial question.", query)
    return {"workflow_status": "oneshot", "final_report": text, "financial_metrics": {}, "retrieved_docs": {}}


def _run_graph(item: dict, *, task_spec_gating: bool, bounded_repair: bool) -> dict:
    root = ROOT / "test_artifacts" / f"product-dev-{uuid4().hex[:8]}"
    config = replace(
        build_test_config(root),
        rag_enabled=False,
        data_mode="live",
        fetch_live_fundamentals=False,
        fetch_sec_fundamentals=False,
        task_spec_gating=task_spec_gating,
        bounded_repair_enabled=bounded_repair,
    )
    app = LumenFinAgentSystem(
        llm_client=LocalFallbackLLMClient(),
        app_config=config,
        market_data_client=FakeMarketDataClient(),
    )
    thread_id = f"pd-{item['id']}"
    started = time.perf_counter()
    result = app.run(item["query"], thread_id=thread_id)
    missing = list(result.get("missing_fields") or [])
    # Catalog items omit an explicit year; fill only the planner time slot.
    if "time_range" in missing and "company" not in missing:
        result = app.resume_with_clarification(thread_id, {"time_range": "FY2025 demo_sample"})
    result["_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["_llm_backend"] = app.llm_client.backend_name
    result["_missing_fields"] = missing
    return result


def summarize(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "n": len(rows),
        "success": round(sum(1 for row in rows if row["score"]["success"]) / n, 4),
        "over_refuse": round(sum(1 for row in rows if row["score"]["over_refuse"]) / n, 4),
        "under_refuse": round(sum(1 for row in rows if row["score"]["under_refuse"]) / n, 4),
        "fatal_data_gap": round(sum(1 for row in rows if row["score"]["fatal_data_gap"]) / n, 4),
        "answer_coverage": round(
            sum(1 for row in rows if str(row["result"].get("workflow_status") or "") == "completed") / n,
            4,
        ),
        "mean_repair_tool_calls": round(
            sum(float(row["result"].get("_repair_tool_calls") or 0) for row in rows) / n,
            3,
        ),
        "citation_support": round(
            sum(float(row["score"]["citation_support"] or 0) for row in rows if row["score"]["citation_support"] is not None)
            / max(1, sum(1 for row in rows if row["score"]["citation_support"] is not None)),
            4,
        ),
        "mean_latency_ms": round(
            sum(float(row["result"].get("_latency_ms") or 0) for row in rows) / n, 1
        ),
        "by_family": {
            family: {
                "n": sum(1 for row in rows if row["item"]["family"] == family),
                "success": round(
                    (
                        sum(
                            1
                            for row in rows
                            if row["item"]["family"] == family and row["score"]["success"]
                        )
                        / max(1, sum(1 for row in rows if row["item"]["family"] == family))
                    ),
                    4,
                ),
            }
            for family in sorted({row["item"]["family"] for row in rows})
        },
    }


def main() -> int:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs" / "product_dev_v1")
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = ROOT / "data" / "eval_product_dev" / "catalog_v1.json"
    dump_catalog_json(catalog_path)
    items = catalog_items(split="dev")
    report: dict = {
        "catalog": str(catalog_path),
        "split": "dev",
        "model": "local-fallback",
        "budget": "offline_only",
        "modes": {},
        "note": (
            "No accuracy promise. Bounded repair stays default-off unless this report "
            "shows a clear benefit without loosening refuse/citation rules."
        ),
    }
    for mode in MODES:
        rows = []
        for item in items:
            started = time.perf_counter()
            if mode == "oneshot":
                result = _oneshot(item["query"])
                result["_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            elif mode == "graph_legacy_fatal":
                result = _run_graph(item, task_spec_gating=False, bounded_repair=False)
            elif mode == "graph_taskspec":
                result = _run_graph(item, task_spec_gating=True, bounded_repair=False)
            else:
                result = _run_graph(item, task_spec_gating=True, bounded_repair=True)
            slim = {
                "workflow_status": result.get("workflow_status"),
                "fatal_data_gap": result.get("fatal_data_gap"),
                "financial_metrics": result.get("financial_metrics") or {},
                "risk_levels": {
                    name: (payload.get("supply_chain") or {}).get("risk_level")
                    for name, payload in (result.get("retrieved_docs") or {}).items()
                },
                "final_report_excerpt": str(result.get("final_report") or "")[:400],
                "_latency_ms": result.get("_latency_ms"),
                "_repair_tool_calls": sum(
                    1
                    for event in (result.get("bounded_repair_trace") or [])
                    if event.get("tool")
                ),
            }
            score = score_item(item, result)
            rows.append({"item": {"id": item["id"], "family": item["family"]}, "score": score, "result": slim})
        report["modes"][mode] = {"summary": summarize(rows), "rows": rows}

    extra_items = [item for item in items if item["family"] in {"doc_risk", "missing_data"}]
    extra_rows: dict[str, list] = {"legacy_no_sample": [], "taskspec_no_sample": []}
    for mode_name, gating in (("legacy_no_sample", False), ("taskspec_no_sample", True)):
        for item in extra_items:
            root = ROOT / "test_artifacts" / f"product-dev-{uuid4().hex[:8]}"
            config = replace(
                build_test_config(root),
                rag_enabled=False,
                data_mode="live",
                fetch_live_fundamentals=False,
                fetch_sec_fundamentals=False,
                task_spec_gating=gating,
                bounded_repair_enabled=False,
            )
            app = LumenFinAgentSystem(
                llm_client=LocalFallbackLLMClient(),
                app_config=config,
                market_data_client=FakeMarketDataClient(),
            )
            thread_id = f"pd-ns-{item['id']}"
            started = time.perf_counter()
            result = app.run(item["query"], thread_id=thread_id)
            missing = list(result.get("missing_fields") or [])
            if "time_range" in missing and "company" not in missing:
                result = app.resume_with_clarification(thread_id, {"time_range": "FY2025 demo_sample"})
            result["_latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
            extra_rows[mode_name].append(
                {
                    "item": {"id": item["id"], "family": item["family"]},
                    "score": score_item(item, result),
                    "result": {
                        "workflow_status": result.get("workflow_status"),
                        "fatal_data_gap": result.get("fatal_data_gap"),
                        "_latency_ms": result.get("_latency_ms"),
                    },
                }
            )
    report["no_sample_risk_slice"] = {
        "note": "Same dev risk/missing items with DATA_MODE=live and sample lookup off. Not a default-path score.",
        "legacy_no_sample": summarize(extra_rows["legacy_no_sample"]),
        "taskspec_no_sample": summarize(extra_rows["taskspec_no_sample"]),
        "rows": extra_rows,
    }
    taskspec = report["modes"]["graph_taskspec"]["summary"]["success"]
    repair = report["modes"]["graph_taskspec_repair"]["summary"]["success"]
    report["repair_beats_taskspec_margin"] = repair > taskspec + 0.05
    # Measured gain is sample_db fill after live retrieval with sample off.
    # Enabling that by default would mix demo fundamentals into live mode.
    report["enable_bounded_repair_default"] = False
    report["enable_bounded_repair_default_reason"] = (
        "Keep MAS_BOUNDED_REPAIR=false. Repair beat TaskSpec only by injecting "
        "SAMPLE_FINANCIAL_DATA into live-empty retrieval; that is an opt-in demo tool, "
        "not a live default. Refuse/citation rules were not loosened."
    )
    (out_dir / "ablation_dev.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# product-dev v1 ablation (dev split, offline LocalFallbackLLM)",
        "",
        f"Catalog: `{catalog_path.as_posix()}`",
        "",
        "| mode | n | success | coverage | over_refuse | under_refuse | fatal_gap | citation | repair_calls | mean_ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        summary = report["modes"][mode]["summary"]
        lines.append(
            f"| {mode} | {summary['n']} | {summary['success']} | {summary['answer_coverage']} | "
            f"{summary['over_refuse']} | {summary['under_refuse']} | {summary['fatal_data_gap']} | "
            f"{summary['citation_support']} | {summary['mean_repair_tool_calls']} | "
            f"{summary['mean_latency_ms']} |"
        )
    ns = report["no_sample_risk_slice"]
    lines.extend(
        [
            "",
            "No-sample risk/missing slice (live mode, sample off):",
            f"- legacy fatal gating: success={ns['legacy_no_sample']['success']} fatal_gap={ns['legacy_no_sample']['fatal_data_gap']}",
            f"- TaskSpec gating: success={ns['taskspec_no_sample']['success']} fatal_gap={ns['taskspec_no_sample']['fatal_data_gap']}",
            "",
        ]
    )
    lines.extend(
        [
            "",
            f"Repair vs TaskSpec success margin >0.05? **{report['repair_beats_taskspec_margin']}**. "
            f"Enable bounded repair by default? **{report['enable_bounded_repair_default']}**. "
            f"{report['enable_bounded_repair_default_reason']}",
            "",
            "This is not an 80%/95% claim. Test split was not scored.",
            "",
        ]
    )
    (out_dir / "ablation_dev.md").write_text("\n".join(lines), encoding="utf-8")
    print((out_dir / "ablation_dev.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
