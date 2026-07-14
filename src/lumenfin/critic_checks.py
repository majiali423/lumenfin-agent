from __future__ import annotations

import re
from typing import Any

from .artifacts import Violation


_DISCLAIMER_PATTERNS = (
    re.compile(r"风险免责声明"),
    re.compile(r"(?:^|\n)\s*\**Disclaimer\**?\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"not investment advice", re.IGNORECASE),
    re.compile(r"does not constitute investment advice", re.IGNORECASE),
)

_PROVENANCE_PATTERNS = (
    re.compile(r"数据来源", re.IGNORECASE),
    re.compile(r"data sources?", re.IGNORECASE),
    re.compile(r"methodology.*data", re.IGNORECASE),
    re.compile(r"uploaded documents?:", re.IGNORECASE),
    re.compile(r"structured_source", re.IGNORECASE),
)


def check_data_completeness(state: dict[str, Any]) -> list[Violation]:
    violations: list[Violation] = []
    companies = state.get("companies") or []
    financial_metrics = state.get("financial_metrics") or {}
    sentiment_analysis = state.get("sentiment_analysis") or {}

    for company in companies:
        name = str(company)
        if name not in financial_metrics:
            violations.append(
                Violation(
                    code="missing_quantitative_results",
                    severity="high",
                    message=f"{name}: missing quantitative results.",
                    company=name,
                    repair_target="quant",
                )
            )
        if name not in sentiment_analysis:
            violations.append(
                Violation(
                    code="missing_sentiment_analysis",
                    severity="medium",
                    message=f"{name}: missing sentiment analysis.",
                    company=name,
                    repair_target="psychologist",
                )
            )
    return violations


def check_retrieval_provenance(state: dict[str, Any]) -> list[Violation]:
    """In live mode, structured fundamentals must not silently fall back to empty payloads."""
    if str(state.get("data_mode") or "demo") != "live":
        return []

    violations: list[Violation] = []
    retrieved_docs = state.get("retrieved_docs") or {}
    for company in state.get("companies") or []:
        name = str(company)
        bundle = retrieved_docs.get(name) or {}
        provenance = bundle.get("provenance") or {}
        structured_source = str(
            provenance.get("structured_source")
            or bundle.get("structured_source")
            or "none"
        )
        confidence = bundle.get("confidence") or {}
        overall = float(confidence.get("overall") or 0.0)

        if structured_source == "none":
            violations.append(
                Violation(
                    code="missing_structured_data",
                    severity="high",
                    message=(
                        f"{name}: live mode requires document-extracted, sec_companyfacts, "
                        f"yahoo_fundamentals, or other verified structured data; got structured_source=none."
                    ),
                    company=name,
                    repair_target="retrieval",
                )
            )
        elif overall < 0.35:
            violations.append(
                Violation(
                    code="low_retrieval_confidence",
                    severity="medium",
                    message=(
                        f"{name}: retrieval confidence {overall:.2f} is below live-mode threshold 0.35."
                    ),
                    company=name,
                    repair_target="retrieval",
                )
            )
    return violations


def check_report_compliance(report_stub: str) -> list[Violation]:
    """Rule-based report checks. Only meaningful when report_sections exist (repair loops)."""
    if not (report_stub or "").strip():
        return []

    violations: list[Violation] = []

    if not any(pattern.search(report_stub) for pattern in _DISCLAIMER_PATTERNS):
        violations.append(
            Violation(
                code="missing_risk_disclaimer",
                severity="high",
                message="Report missing risk disclaimer markers.",
                repair_target="retrieval",
            )
        )

    if not any(pattern.search(report_stub) for pattern in _PROVENANCE_PATTERNS):
        violations.append(
            Violation(
                code="missing_data_provenance",
                severity="medium",
                message="Report missing data provenance markers.",
                repair_target="retrieval",
            )
        )
    return violations


def run_critic_checks(state: dict[str, Any]) -> list[Violation]:
    violations: list[Violation] = []
    violations.extend(check_data_completeness(state))
    violations.extend(check_retrieval_provenance(state))
    report_stub = "\n".join(state.get("report_sections") or [])
    violations.extend(check_report_compliance(report_stub))
    return violations


def violations_to_messages(violations: list[Violation]) -> list[str]:
    return [violation.message for violation in violations]
