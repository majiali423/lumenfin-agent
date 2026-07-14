from __future__ import annotations

from .critic_checks import violations_to_messages
from .repair_policies import resolve_repair_target, resolve_repair_target_from_codes
from .artifacts import Violation


def classify_critic_repair_target(findings: list[str]) -> str:
    """Backward-compatible wrapper for legacy string findings."""
    joined = " ".join(findings).lower()
    codes: list[str] = []
    if "missing quantitative" in joined or "quantitative results" in joined:
        codes.append("missing_quantitative_results")
    if "missing sentiment" in joined:
        codes.append("missing_sentiment_analysis")
    if "structured_source=none" in joined or "missing structured" in joined:
        codes.append("missing_structured_data")
    if "retrieval confidence" in joined:
        codes.append("low_retrieval_confidence")
    if "缺少风险免责声明" in joined or "missing risk disclaimer" in joined:
        codes.append("missing_risk_disclaimer")
    if "缺少数据来源" in joined or "missing data provenance" in joined:
        codes.append("missing_data_provenance")
    if codes:
        return resolve_repair_target_from_codes(codes)
    return "retrieval"


def classify_critic_violations(violations: list[Violation]) -> str:
    return resolve_repair_target(violations)


def compliance_messages(violations: list[Violation]) -> list[str]:
    return violations_to_messages(violations)
