from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_STEPS = ("query_planner", "supervisor", "retrieval", "quant", "psychologist", "critic", "synthesizer")
REQUIRED_REPORT_MARKERS = (
    "Executive Summary",
    "Financial Performance Analysis",
    "Risk",
    "Compliance",
    "Methodology",
    "Disclaimer",
)


@dataclass(frozen=True)
class EvaluationResult:
    score: int
    grade: str
    checks: dict[str, dict[str, Any]]
    recommendations: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "grade": self.grade,
            "checks": self.checks,
            "recommendations": self.recommendations,
        }


def evaluate_run_state(state: dict[str, Any]) -> EvaluationResult:
    checks = {
        "pipeline_completeness": _check_pipeline_completeness(state),
        "report_contract": _check_report_contract(state),
        "evidence_grounding": _check_evidence_grounding(state),
        "operational_reliability": _check_operational_reliability(state),
    }
    score = round(sum(item["score"] for item in checks.values()) / len(checks))
    grade = _grade(score)
    recommendations = _build_recommendations(checks)
    return EvaluationResult(score=score, grade=grade, checks=checks, recommendations=recommendations)


def _check_pipeline_completeness(state: dict[str, Any]) -> dict[str, Any]:
    audit_log = state.get("audit_log", [])
    observed_steps = [event.get("step") for event in audit_log]
    missing = [step for step in REQUIRED_STEPS if step not in observed_steps]
    blocked = [
        event.get("step")
        for event in audit_log
        if event.get("status") in {"blocked", "failed", "error"}
    ]
    score = 100
    score -= len(missing) * 15
    score -= len(blocked) * 10
    return {
        "score": max(0, score),
        "passed": not missing and not blocked,
        "missing_steps": missing,
        "blocked_steps": blocked,
        "observed_steps": observed_steps,
    }


def _check_report_contract(state: dict[str, Any]) -> dict[str, Any]:
    report = state.get("final_report", "")
    missing_markers = [marker for marker in REQUIRED_REPORT_MARKERS if marker not in report]
    length_ok = len(report) >= 2500
    score = 100
    score -= len(missing_markers) * 12
    if not length_ok:
        score -= 20
    return {
        "score": max(0, score),
        "passed": not missing_markers and length_ok,
        "missing_markers": missing_markers,
        "character_count": len(report),
    }


def _check_evidence_grounding(state: dict[str, Any]) -> dict[str, Any]:
    companies = state.get("companies", [])
    retrieved_docs = state.get("retrieved_docs", {})
    financial_metrics = state.get("financial_metrics", {})
    risk_scores = state.get("risk_scores", {})
    sentiment = state.get("sentiment_analysis", {})

    company_coverage = {}
    for company in companies:
        company_coverage[company] = {
            "retrieval": bool(retrieved_docs.get(company)),
            "metrics": bool(financial_metrics.get(company)),
            "risk": bool(risk_scores.get(company)),
            "sentiment": bool(sentiment.get(company)),
        }

    total_slots = max(1, len(companies) * 4)
    filled_slots = sum(
        1
        for coverage in company_coverage.values()
        for present in coverage.values()
        if present
    )
    score = round(filled_slots / total_slots * 100)
    return {
        "score": score,
        "passed": score >= 85,
        "company_coverage": company_coverage,
        "llm_backend": state.get("llm_backend", "unknown"),
    }


def _check_operational_reliability(state: dict[str, Any]) -> dict[str, Any]:
    degraded = bool(state.get("degraded_mode"))
    findings = state.get("compliance_findings", [])
    replan_reason = state.get("replan_reason")
    artifacts_ready = bool(state.get("final_report")) and bool(state.get("audit_log"))

    score = 100
    if degraded:
        score -= 25
    if findings:
        score -= min(30, len(findings) * 8)
    if replan_reason:
        score -= 10
    if not artifacts_ready:
        score -= 25
    return {
        "score": max(0, score),
        "passed": score >= 80,
        "degraded_mode": degraded,
        "compliance_findings": findings,
        "open_replan_reason": replan_reason,
        "artifacts_ready": artifacts_ready,
    }


def _grade(score: int) -> str:
    if score >= 90:
        return "production-ready demo"
    if score >= 75:
        return "portfolio-ready with caveats"
    if score >= 60:
        return "needs reliability work"
    return "not ready"


def _build_recommendations(checks: dict[str, dict[str, Any]]) -> list[str]:
    recommendations: list[str] = []
    pipeline = checks["pipeline_completeness"]
    if pipeline["missing_steps"]:
        recommendations.append(f"Add or repair missing agent steps: {', '.join(pipeline['missing_steps'])}.")
    if pipeline["blocked_steps"]:
        recommendations.append(f"Investigate blocked agent steps: {', '.join(pipeline['blocked_steps'])}.")

    report = checks["report_contract"]
    if report["missing_markers"]:
        recommendations.append(f"Restore required report sections: {', '.join(report['missing_markers'])}.")
    if report["character_count"] < 2500:
        recommendations.append("Increase report depth; the final report is too short for an analyst workflow.")

    evidence = checks["evidence_grounding"]
    if evidence["score"] < 85:
        recommendations.append("Improve per-company evidence coverage across retrieval, metrics, risk, and sentiment.")

    reliability = checks["operational_reliability"]
    if reliability["degraded_mode"]:
        recommendations.append("Explain degraded-mode behavior in the UI and add a recovery path.")
    if reliability["compliance_findings"]:
        recommendations.append("Resolve compliance findings before presenting the report as client-facing.")
    if not recommendations:
        recommendations.append("Keep this trace as a golden run for regression testing after prompt or tool changes.")
    return recommendations
