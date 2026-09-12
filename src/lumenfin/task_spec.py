"""Task sufficiency: what this query is allowed to answer without inventing numbers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .query_focus import RATIO_METRICS, skip_enrichment_for_plan

RATIO_DIMENSIONS = frozenset({"profitability", "r_and_d", "liquidity", "solvency", "market"})
NARRATIVE_DIMENSIONS = frozenset({"supply_chain", "sentiment", "compliance", "document_evidence"})
ALLOWED_REPAIR_TOOLS = ("sample_fundamentals", "safe_ratio", "stop")


@dataclass(frozen=True)
class TaskSpec:
    intent: str
    dimensions: tuple[str, ...]
    requires_ast_ratios: bool
    skip_quant: bool
    allow_risk_without_ratios: bool
    skip_enrichment: bool = False
    requested_metrics: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ALLOWED_REPAIR_TOOLS
    max_repair_steps: int = 2
    max_tool_calls: int = 3

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = list(self.dimensions)
        payload["allowed_tools"] = list(self.allowed_tools)
        return payload


def task_spec_from_plan(
    plan: dict[str, Any] | None,
    *,
    max_repair_steps: int = 2,
    max_tool_calls: int = 3,
) -> TaskSpec:
    plan = plan or {}
    intent = str(plan.get("intent") or "financial_diligence")
    dims = tuple(str(item) for item in (plan.get("analysis_dimensions") or []) if str(item).strip())
    dim_set = set(dims)
    requested = tuple(str(item) for item in (plan.get("requested_metrics") or ()) if str(item).strip())
    narrative_only = bool(dim_set) and dim_set.issubset(NARRATIVE_DIMENSIONS)
    ratio_asked = any(item in RATIO_METRICS for item in requested)
    if intent == "risk_compliance_review":
        # Default planner dimensions still include profitability. Risk answers
        # must not be fail-closed on missing AST inputs.
        requires = False
    elif requested and not ratio_asked:
        requires = False
    elif narrative_only and intent != "comparative_financial_diligence":
        requires = False
    else:
        requires = True
    skip_enrichment = skip_enrichment_for_plan({**plan, "requested_metrics": list(requested)})
    skip_quant = (not requires) and "document_evidence" not in dim_set
    if ratio_asked or (requested and not skip_enrichment):
        skip_quant = False
    if skip_enrichment and not ratio_asked:
        skip_quant = True
    return TaskSpec(
        intent=intent,
        dimensions=dims,
        requires_ast_ratios=requires,
        skip_quant=skip_quant,
        allow_risk_without_ratios=not requires,
        skip_enrichment=skip_enrichment,
        requested_metrics=requested,
        max_repair_steps=max(1, int(max_repair_steps)),
        max_tool_calls=max(1, int(max_tool_calls)),
    )


def task_spec_from_state(state: dict[str, Any] | None) -> TaskSpec:
    state = state or {}
    existing = state.get("task_spec")
    if isinstance(existing, dict) and "requires_ast_ratios" in existing:
        return TaskSpec(
            intent=str(existing.get("intent") or ""),
            dimensions=tuple(existing.get("dimensions") or ()),
            requires_ast_ratios=bool(existing.get("requires_ast_ratios")),
            skip_quant=bool(existing.get("skip_quant")),
            allow_risk_without_ratios=bool(existing.get("allow_risk_without_ratios")),
            skip_enrichment=bool(existing.get("skip_enrichment")),
            requested_metrics=tuple(existing.get("requested_metrics") or ()),
            allowed_tools=tuple(existing.get("allowed_tools") or ALLOWED_REPAIR_TOOLS),
            max_repair_steps=int(existing.get("max_repair_steps") or 2),
            max_tool_calls=int(existing.get("max_tool_calls") or 3),
        )
    return task_spec_from_plan(state.get("query_plan") if isinstance(state.get("query_plan"), dict) else {})
