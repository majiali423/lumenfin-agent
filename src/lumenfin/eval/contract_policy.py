"""Versioned document-task scoring policy: gold vs FinAgentBench applicability.

Does not change product generation. Expected subjects come from the eval task
contract, never from model output, and never default to NVIDIA.
"""

from __future__ import annotations

import re
from typing import Any

SCORING_POLICY_VERSION = "lumenfin_eval_contract.v1"

STATUS_PASSED = "passed"
STATUS_FAILED = "failed"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"
STATUS_UNDETERMINED = "undetermined"

LANE_SHARED = "shared_export"
LANE_INTERNAL = "b2_internal"

FAMILY_FACT = "financial_fact"
FAMILY_RATIO = "ratio_calc"
FAMILY_PERIOD = "multipage_period"
FAMILY_COMPARE = "comparison_scope"
FAMILY_RISK = "evidenced_risk"
FAMILY_REFUSE = "refuse_or_clarify"

ANSWER_FAMILIES = {FAMILY_FACT, FAMILY_RATIO, FAMILY_PERIOD, FAMILY_COMPARE}

# What each family is supposed to check. Used by docs and tests; not case-id logic.
FAMILY_CHECK_MATRIX: dict[str, dict[str, Any]] = {
    FAMILY_FACT: {
        "gold": (
            "source facts: value, entity, period, unit, citation/source; "
            "task complete when every required company+metric+period is answered"
        ),
        "bench_shared": ("visible_supported_claims",),
        "bench_internal": ("entity_coverage",),
        "not_claimed": ("full free-text semantics",),
    },
    FAMILY_RATIO: {
        "gold": (
            "same fact identity as financial_fact, plus computed ratio/percent "
            "against operands or gold with allowed display precision"
        ),
        "bench_shared": ("visible_supported_claims",),
        "bench_internal": ("entity_coverage", "numeric_correctness", "unit_currency_consistency"),
        "not_claimed": ("gold-independent formula invention",),
    },
    FAMILY_PERIOD: {
        "gold": "same source-fact identity; periods must match the labeled field year",
        "bench_shared": ("visible_supported_claims",),
        "bench_internal": ("entity_coverage",),
        "not_claimed": ("unparsed narrative time phrases beyond gold labels",),
    },
    FAMILY_COMPARE: {
        "gold": (
            "each required company+metric+period fact; naming a company is not completeness"
        ),
        "bench_shared": ("visible_supported_claims",),
        "bench_internal": ("entity_coverage", "entity_leakage"),
        "not_claimed": ("ranking quality beyond stated facts",),
    },
    FAMILY_REFUSE: {
        "gold": "action matches refuse/clarify; forbidden numeric claims still fail",
        "bench_shared": ("visible_supported_claims when the output still asserts amounts",),
        "bench_internal": (),
        "not_claimed": ("missing checkable numbers is not a gold fail for a correct refuse",),
    },
    FAMILY_RISK: {
        "gold": "required risk points and evidence; missing finance numbers is not a fail",
        "bench_shared": ("visible_supported_claims only if the output asserts amounts",),
        "bench_internal": (),
        "not_claimed": ("FinAgentBench does not judge full risk-analysis semantics",),
    },
}

_FILENAME_RE = re.compile(r"[\w.\-]+\.(?:pdf|txt|html|md)(?:#p\d+)?", re.IGNORECASE)
_AMOUNT_RE = re.compile(
    r"(?P<num>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*"
    r"(?P<unit>billion|million|percent|%|bn|mm)\b",
    re.IGNORECASE,
)
_METRICISH = re.compile(
    r"operating income|operating margin|revenue|ebitda|r&d|r and d|research and development",
    re.IGNORECASE,
)


def output_has_numeric_assertions(text: str) -> bool:
    """True when the visible answer asserts a financial amount, not merely a page id."""
    blob = _FILENAME_RE.sub(" ", str(text or ""))
    blob = re.sub(r"\[[^\]]*\]", " ", blob)
    if _AMOUNT_RE.search(blob):
        return True
    if re.search(r"\b\d+\.\d{2,}\b", blob) and _METRICISH.search(blob):
        return True
    return False


def expected_subjects(task: dict[str, Any]) -> list[str]:
    """Subjects from the eval-side task contract. Empty means missing, not NVIDIA."""
    contract = task.get("scoring_contract") if isinstance(task.get("scoring_contract"), dict) else {}
    if "expected_entities" in contract:
        names: list[str] = []
        for item in contract.get("expected_entities") or []:
            value = str(item or "").strip()
            if value and value not in names:
                names.append(value)
        return names
    names = []
    for fact in task.get("required_facts") or []:
        if not isinstance(fact, dict):
            continue
        entity = str(fact.get("entity") or "").strip()
        if entity and entity not in names:
            names.append(entity)
    if names:
        return names
    query = str(task.get("query") or "")
    if not query.strip():
        return []
    from lumenfin.tools import extract_companies_from_query

    return extract_companies_from_query(query, document_contexts=None, llm_client=None)


def subjects_required(task: dict[str, Any]) -> bool:
    family = str(task.get("family") or "")
    action = str(task.get("expected_action") or "answer")
    if action != "answer":
        return False
    if family == FAMILY_COMPARE:
        return True
    if family in ANSWER_FAMILIES and (task.get("required_facts") or []):
        return True
    return False


def missing_subjects_error(task: dict[str, Any]) -> str | None:
    if not subjects_required(task):
        return None
    if expected_subjects(task):
        return None
    return (
        "expected_subjects_missing: task contract has no required-fact entities, "
        "scoring_contract.expected_entities, or named companies in the query; "
        "refusing to default a subject"
    )


def forbidden_subjects(task: dict[str, Any]) -> list[str]:
    expected = {item.lower() for item in expected_subjects(task)}
    names: list[str] = []
    for banned in task.get("forbidden_claims") or []:
        if not isinstance(banned, dict):
            continue
        entity = str(banned.get("entity") or "").strip()
        if entity and entity.lower() not in expected and entity not in names:
            names.append(entity)
    return names


def plan_contract_checks(
    task: dict[str, Any],
    *,
    final_output: str,
    has_finrun: bool,
    system: str | None,
) -> dict[str, Any]:
    """Decide which Bench metrics to run. Does not execute them."""
    family = str(task.get("family") or "")
    action = str(task.get("expected_action") or "answer")
    system_name = str(system or ("b2_agent" if has_finrun else "b1_rag"))
    is_b2 = system_name.startswith("b2")
    has_numbers = output_has_numeric_assertions(final_output)
    subject_error = missing_subjects_error(task)
    subjects = expected_subjects(task)
    na: list[dict[str, str]] = []
    shared: list[str] = []
    internal: list[str] = []

    def _na(name: str, reason: str) -> None:
        na.append({"name": name, "reason": reason, "lane": LANE_SHARED if name == "visible_supported_claims" else LANE_INTERNAL})

    export_required = is_b2
    if not has_finrun:
        if export_required:
            pass  # caller marks applicable metrics unavailable
        else:
            _na("visible_supported_claims", "b1_has_no_agent_export")
            _na("entity_coverage", "b1_internal_structure_not_required")
            _na("entity_leakage", "b1_internal_structure_not_required")
            _na("numeric_correctness", "b1_internal_structure_not_required")
            _na("unit_currency_consistency", "b1_internal_structure_not_required")
            return {
                "scoring_policy_version": SCORING_POLICY_VERSION,
                "system": system_name,
                "family": family,
                "expected_entities": subjects,
                "forbidden_entities": forbidden_subjects(task),
                "subjects_error": subject_error,
                "shared_metrics": [],
                "internal_metrics": [],
                "not_applicable": na,
                "require_checkable_internal": False,
                "export_required": False,
                "has_numeric_assertions": has_numbers,
            }

    vsc_applicable = False
    if family == FAMILY_RISK:
        if has_numbers:
            vsc_applicable = True
        else:
            _na("visible_supported_claims", "risk_task_without_numeric_assertions")
    elif family == FAMILY_REFUSE or action in {"refuse", "clarify"}:
        if has_numbers:
            vsc_applicable = True
        else:
            _na("visible_supported_claims", "refuse_or_clarify_without_numeric_assertions")
    elif family in ANSWER_FAMILIES and action in {"answer", "answer_or_clarify"}:
        vsc_applicable = True
    else:
        if has_numbers:
            vsc_applicable = True
        else:
            _na("visible_supported_claims", "no_numeric_assertions_for_family")

    if vsc_applicable:
        shared.append("visible_supported_claims")

    if is_b2 and family in ANSWER_FAMILIES and action in {"answer", "answer_or_clarify"}:
        if subject_error:
            pass
        elif subjects:
            internal.append("entity_coverage")
        else:
            _na("entity_coverage", "no_expected_subjects_on_non_required_task")
    else:
        _na("entity_coverage", "not_applicable_for_family_or_b1")

    if is_b2 and family == FAMILY_COMPARE and forbidden_subjects(task):
        internal.append("entity_leakage")
    else:
        _na("entity_leakage", "no_forbidden_peer_contract" if is_b2 else "b1_internal_structure_not_required")

    if is_b2 and family == FAMILY_RATIO and action in {"answer", "answer_or_clarify"}:
        internal.append("numeric_correctness")
        internal.append("unit_currency_consistency")
    else:
        _na("numeric_correctness", "not_a_ratio_export_or_not_b2")
        _na("unit_currency_consistency", "not_a_ratio_export_or_not_b2")

    return {
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "system": system_name,
        "family": family,
        "expected_entities": subjects,
        "forbidden_entities": forbidden_subjects(task),
        "subjects_error": subject_error,
        "shared_metrics": shared,
        "internal_metrics": internal,
        "not_applicable": na,
        "require_checkable_internal": FAMILY_RATIO == family and "numeric_correctness" in internal,
        "export_required": export_required,
        "has_numeric_assertions": has_numbers,
    }


def check_record(
    *,
    name: str,
    lane: str,
    status: str,
    passed: bool | None,
    findings: list[str],
    applicable_reason: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "lane": lane,
        "status": status,
        "passed": passed,
        "findings": findings,
        "applicable_reason": applicable_reason,
    }


def rollup_status(checks: list[dict[str, Any]]) -> str:
    if not checks:
        return STATUS_NOT_APPLICABLE
    statuses = [str(item.get("status") or "") for item in checks]
    if any(status == STATUS_ERROR for status in statuses):
        return STATUS_ERROR
    if any(status == STATUS_UNAVAILABLE for status in statuses):
        return STATUS_UNAVAILABLE
    if any(status == STATUS_UNDETERMINED for status in statuses):
        return STATUS_UNDETERMINED
    executed = [status for status in statuses if status != STATUS_NOT_APPLICABLE]
    if not executed:
        return STATUS_NOT_APPLICABLE
    if any(status == STATUS_FAILED for status in executed):
        return STATUS_FAILED
    if all(status == STATUS_PASSED for status in executed):
        return STATUS_PASSED
    return STATUS_UNDETERMINED


def development_acceptance(*, task_passed: bool, contract_status: str) -> bool:
    """New field. Gold pass plus applicable Bench not failed/unavailable/error.

    Does not change diagnostic_pass. B2-only internal checks are excluded.
    """
    if not task_passed:
        return False
    return contract_status in {STATUS_PASSED, STATUS_NOT_APPLICABLE}
