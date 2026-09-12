"""Run applicable FinAgentBench metrics through evaluate_run."""

from __future__ import annotations

from typing import Any

from lumenfin.eval.contract_policy import (
    LANE_INTERNAL,
    LANE_SHARED,
    SCORING_POLICY_VERSION,
    STATUS_ERROR,
    STATUS_FAILED,
    STATUS_NOT_APPLICABLE,
    STATUS_PASSED,
    STATUS_UNAVAILABLE,
    check_record,
    development_acceptance,
    plan_contract_checks,
    rollup_status,
)


def evaluate_contract(
    task: dict[str, Any],
    *,
    final_output: str,
    finrun: dict[str, Any] | None,
    system: str | None = None,
) -> dict[str, Any]:
    plan = plan_contract_checks(
        task,
        final_output=final_output,
        has_finrun=isinstance(finrun, dict),
        system=system,
    )
    checks: list[dict[str, Any]] = []
    for item in plan["not_applicable"]:
        checks.append(
            check_record(
                name=str(item["name"]),
                lane=str(item.get("lane") or LANE_SHARED),
                status=STATUS_NOT_APPLICABLE,
                passed=None,
                findings=[],
                applicable_reason=str(item.get("reason") or "not_applicable"),
            )
        )

    subjects_error = plan.get("subjects_error")
    if subjects_error:
        checks.append(
            check_record(
                name="expected_subjects",
                lane=LANE_SHARED,
                status=STATUS_ERROR,
                passed=False,
                findings=[str(subjects_error)],
                applicable_reason="task_contract_requires_subjects",
            )
        )

    needed = list(plan["shared_metrics"]) + list(plan["internal_metrics"])
    if needed and plan["export_required"] and not isinstance(finrun, dict):
        for name in needed:
            lane = LANE_SHARED if name in plan["shared_metrics"] else LANE_INTERNAL
            checks.append(
                check_record(
                    name=name,
                    lane=lane,
                    status=STATUS_UNAVAILABLE,
                    passed=False,
                    findings=["finrun_export_missing"],
                    applicable_reason="b2_export_required",
                )
            )
        return _bundle(plan, checks, runner="skipped_missing_finrun", report=None)

    if not needed:
        return _bundle(plan, checks, runner="not_applicable", report=None)

    try:
        from finagentbench.runner import evaluate_run
        from finagentbench.schema import ValidationError
    except Exception as exc:
        for name in needed:
            lane = LANE_SHARED if name in plan["shared_metrics"] else LANE_INTERNAL
            checks.append(
                check_record(
                    name=name,
                    lane=lane,
                    status=STATUS_UNAVAILABLE,
                    passed=False,
                    findings=[f"finagentbench_unavailable:{type(exc).__name__}"],
                    applicable_reason="runner_dependency_missing",
                )
            )
        return _bundle(plan, checks, runner="unavailable", report=None)

    run_payload = _finrun_for_scoring(finrun, final_output)
    reports: list[dict[str, Any]] = []
    try:
        if plan["shared_metrics"]:
            report = evaluate_run(
                run_payload,
                _case(
                    plan,
                    plan["shared_metrics"],
                    require_checkable=bool(plan.get("has_numeric_assertions")),
                    system=plan["system"],
                ),
            )
            reports.append(_report_dict(report, plan["shared_metrics"]))
            _extend_from_report(checks, report, set(plan["shared_metrics"]), LANE_SHARED)
        if plan["internal_metrics"]:
            report = evaluate_run(
                run_payload,
                _case(
                    plan,
                    plan["internal_metrics"],
                    require_checkable=bool(plan["require_checkable_internal"]),
                    system=plan["system"],
                ),
            )
            reports.append(_report_dict(report, plan["internal_metrics"]))
            _extend_from_report(checks, report, set(plan["internal_metrics"]), LANE_INTERNAL)
    except ValidationError as exc:
        for name in needed:
            lane = LANE_SHARED if name in plan["shared_metrics"] else LANE_INTERNAL
            if any(item["name"] == name and item["status"] not in {STATUS_NOT_APPLICABLE} for item in checks):
                continue
            checks.append(
                check_record(
                    name=name,
                    lane=lane,
                    status=STATUS_ERROR,
                    passed=False,
                    findings=[f"finrun_or_case_invalid:{exc}"],
                    applicable_reason="evaluate_run_validation_error",
                )
            )
        return _bundle(plan, checks, runner="evaluate_run", report={"error": str(exc)})
    except Exception as exc:
        for name in needed:
            lane = LANE_SHARED if name in plan["shared_metrics"] else LANE_INTERNAL
            if any(
                item["name"] == name
                and item["lane"] == lane
                and item["status"] not in {STATUS_NOT_APPLICABLE}
                for item in checks
            ):
                continue
            checks.append(
                check_record(
                    name=name,
                    lane=lane,
                    status=STATUS_ERROR,
                    passed=False,
                    findings=[f"evaluate_run_failed:{type(exc).__name__}:{exc}"],
                    applicable_reason="evaluate_run_exception",
                )
            )
        return _bundle(plan, checks, runner="evaluate_run", report={"error": str(exc)})

    return _bundle(plan, checks, runner="finagentbench.evaluate_run", report=reports)


def _case(
    plan: dict[str, Any],
    enabled: list[str],
    *,
    require_checkable: bool,
    system: str,
) -> dict[str, Any]:
    path = "product_workflow" if str(system).startswith("b2") else "retrieval_qa"
    return {
        "expected_entities": list(plan.get("expected_entities") or []),
        "forbidden_entities": list(plan.get("forbidden_entities") or []),
        "required_steps": [],
        "enabled_metrics": list(enabled),
        "scoring_version": "3",
        "execution_path": path,
        "case_mode": "quality",
        "require_checkable_metrics": require_checkable,
        "require_visible_claim_citations": False,
        "numeric_tolerance": 0.01,
        "block_on_severity": ["high", "critical"],
    }


def _finrun_for_scoring(finrun: dict[str, Any], final_output: str) -> dict[str, Any]:
    payload = dict(finrun)
    payload["final_output"] = str(final_output or payload.get("final_output") or "")
    return payload


def _extend_from_report(checks: list[dict[str, Any]], report: Any, names: set[str], lane: str) -> None:
    for metric in report.metrics:
        if metric.name not in names:
            continue
        failed = [item for item in metric.findings if item.severity in {"high", "critical"}]
        status = STATUS_PASSED if metric.passed and not failed else STATUS_FAILED
        checks.append(
            check_record(
                name=metric.name,
                lane=lane,
                status=status,
                passed=status == STATUS_PASSED,
                findings=[item.message for item in metric.findings],
                applicable_reason="evaluate_run",
            )
        )


def _report_dict(report: Any, enabled: list[str]) -> dict[str, Any]:
    return {
        "run_id": report.run_id,
        "score": report.score,
        "passed": report.passed,
        "enabled_metrics": list(enabled),
        "scoring_version": report.scoring_version,
        "metrics": [
            {"name": item.name, "score": item.score, "passed": item.passed}
            for item in report.metrics
        ],
    }


def _bundle(
    plan: dict[str, Any],
    checks: list[dict[str, Any]],
    *,
    runner: str,
    report: Any,
) -> dict[str, Any]:
    shared = [item for item in checks if item.get("lane") == LANE_SHARED]
    internal = [item for item in checks if item.get("lane") == LANE_INTERNAL]
    contract_status = rollup_status(shared)
    internal_status = rollup_status(internal)
    return {
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "runner": runner,
        "plan": {
            "family": plan.get("family"),
            "system": plan.get("system"),
            "shared_metrics": plan.get("shared_metrics"),
            "internal_metrics": plan.get("internal_metrics"),
            "expected_entities": plan.get("expected_entities"),
            "has_numeric_assertions": plan.get("has_numeric_assertions"),
        },
        "contract_result": {
            "status": contract_status,
            "passed": contract_status == STATUS_PASSED,
            "checks": shared,
        },
        "b2_internal_result": {
            "status": internal_status,
            "passed": internal_status == STATUS_PASSED,
            "checks": internal,
            "note": (
                "B2-only export/structure checks. Not mixed into B1 task_result "
                "or diagnostic_pass."
            ),
        },
        "findings": [
            {
                "name": item["name"],
                "lane": item["lane"],
                "status": item["status"],
                "reason": item.get("applicable_reason"),
                "detail": item.get("findings") or [],
            }
            for item in checks
        ],
        "report": report,
        "eval_acceptance_v1_contract_ok": contract_status in {STATUS_PASSED, STATUS_NOT_APPLICABLE},
    }


def acceptance_from_layers(*, task_passed: bool, contract: dict[str, Any]) -> bool:
    status = str((contract.get("contract_result") or {}).get("status") or "")
    return development_acceptance(task_passed=task_passed, contract_status=status)
