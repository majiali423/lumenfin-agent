"""Bounded gap-fill: propose gap → whitelist tool → verify → answer or stop.

Does not execute model-authored code. Tools are sample lookup and SafeExpressionEvaluator.
"""

from __future__ import annotations

import re
import time
from typing import Any

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .quant_contract import has_computable_fundamentals
from .safe_formula import safe_execute_formula
from .task_spec import ALLOWED_REPAIR_TOOLS, task_spec_from_state

_RATIO_FORMULAS = {
    "ebitda_margin": ("ebitda / revenue", ("ebitda", "revenue")),
    "operating_margin": ("operating_income / revenue", ("operating_income", "revenue")),
    "r_and_d_intensity": ("r_and_d / revenue", ("r_and_d", "revenue")),
}
_SAMPLE_FISCAL_YEAR = "2025"
_YEAR_RE = re.compile(r"(?:FY\s*)?(20\d{2})", re.IGNORECASE)


def sample_fill_allowed(state: dict[str, Any], company: str, *, allow_sample_data: bool) -> tuple[bool, str]:
    """Repair may attempt a fill; sample_db still needs an independent data-source permit."""
    if not allow_sample_data:
        return False, "sample_not_permitted"
    if str(state.get("data_mode") or "").strip().lower() == "live":
        return False, "live_mode_forbids_sample"
    if company not in SAMPLE_FINANCIAL_DATA:
        return False, "company_not_in_sample"
    companies = [str(item) for item in (state.get("companies") or []) if str(item).strip()]
    if companies and company not in companies:
        return False, "company_out_of_scope"
    plan = state.get("query_plan") if isinstance(state.get("query_plan"), dict) else {}
    if bool(plan.get("prefer_uploaded_only")):
        return False, "uploaded_only"
    if state.get("document_contexts"):
        return False, "uploaded_only"
    scope = str((state.get("user_clarification") or {}).get("company_scope") or "").lower()
    if scope == "uploaded":
        return False, "uploaded_only"
    years = _requested_years(state)
    if years:
        extra = years - {_SAMPLE_FISCAL_YEAR}
        if extra or _SAMPLE_FISCAL_YEAR not in years:
            return False, "period_mismatch"
    return True, "ok"


def _requested_years(state: dict[str, Any]) -> set[str]:
    chunks: list[str] = []
    plan = state.get("query_plan") if isinstance(state.get("query_plan"), dict) else {}
    for raw in (plan.get("time_range"), plan.get("period"), state.get("time_range"), state.get("query")):
        if raw:
            chunks.append(str(raw))
    for year in state.get("years") or []:
        chunks.append(str(year))
    found = {match.group(1) for chunk in chunks for match in _YEAR_RE.finditer(chunk)}
    return found


def run_bounded_repair(state: dict[str, Any], *, allow_sample_data: bool, deadline_seconds: float = 2.0) -> dict[str, Any]:
    spec = task_spec_from_state(state)
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []
    retrieved = dict(state.get("retrieved_docs") or {})
    tool_calls = 0
    steps = 0

    def remaining() -> float:
        return deadline_seconds - (time.perf_counter() - started)

    while steps < spec.max_repair_steps and tool_calls < spec.max_tool_calls and remaining() > 0:
        steps += 1
        gap_companies = [
            name
            for name, payload in retrieved.items()
            if not has_computable_fundamentals(payload)
        ]
        if not gap_companies:
            trace.append({"step": steps, "decision": "stop", "reason": "fundamentals_computable"})
            break
        company = gap_companies[0]
        if "sample_fundamentals" in spec.allowed_tools:
            allowed, reason = sample_fill_allowed(state, company, allow_sample_data=allow_sample_data)
            if allowed:
                tool_calls += 1
                sample = SAMPLE_FINANCIAL_DATA[company]
                payload = dict(retrieved.get(company) or {})
                payload["market_data"] = dict(sample.get("market_data") or {})
                payload.setdefault("supply_chain", sample.get("supply_chain") or {})
                payload.setdefault("earnings_call_quotes", sample.get("earnings_call_quotes") or [])
                payload["structured_source"] = "sample_db"
                retrieved[company] = payload
                trace.append(
                    {
                        "step": steps,
                        "decision": "tool",
                        "tool": "sample_fundamentals",
                        "input": {"company": company},
                        "output": {
                            "structured_source": "sample_db",
                            "computable": has_computable_fundamentals(payload),
                        },
                    }
                )
                continue
            trace.append(
                {
                    "step": steps,
                    "decision": "skip_sample",
                    "reason": reason,
                    "company": company,
                }
            )
        if "safe_ratio" in spec.allowed_tools:
            payload = retrieved.get(company) or {}
            market = dict(payload.get("market_data") or {})
            computed: dict[str, float] = {}
            for name, (formula, keys) in _RATIO_FORMULAS.items():
                if any(market.get(key) in (None, 0) for key in keys):
                    continue
                try:
                    computed[name] = safe_execute_formula(
                        formula, {key: float(market[key]) for key in keys}
                    )
                except (KeyError, ValueError, ZeroDivisionError):
                    continue
            tool_calls += 1
            trace.append(
                {
                    "step": steps,
                    "decision": "tool",
                    "tool": "safe_ratio",
                    "input": {"company": company, "keys": sorted(market)},
                    "output": computed,
                }
            )
            if computed:
                metrics = dict(state.get("financial_metrics") or {})
                metrics[company] = {**(metrics.get(company) or {}), **computed}
                state = {**state, "financial_metrics": metrics}
            trace.append({"step": steps, "decision": "stop", "reason": "safe_ratio_exhausted"})
            break
        trace.append({"step": steps, "decision": "stop", "reason": "no_allowed_tool"})
        break
    else:
        if not trace or trace[-1].get("decision") != "stop":
            trace.append(
                {
                    "step": steps,
                    "decision": "stop",
                    "reason": "budget_exhausted",
                    "tool_calls": tool_calls,
                }
            )

    computable = [name for name, payload in retrieved.items() if has_computable_fundamentals(payload)]
    missing = bool(retrieved) and not computable
    fatal = missing and spec.requires_ast_ratios
    update: dict[str, Any] = {
        "retrieved_docs": retrieved,
        "task_spec": spec.to_dict(),
        "bounded_repair_trace": trace,
        "fatal_data_gap": fatal,
        "degraded_mode": fatal or bool(state.get("degraded_mode")),
        "data_gap_detail": (
            (state.get("data_gap_detail") or "")
            if fatal
            else ("" if computable else str(state.get("data_gap_detail") or "narrative_only"))
        ),
    }
    if state.get("financial_metrics"):
        update["financial_metrics"] = state["financial_metrics"]
    return update
