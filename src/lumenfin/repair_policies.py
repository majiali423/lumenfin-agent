from __future__ import annotations

from .artifacts import RepairPolicy, Violation


REPAIR_POLICIES: tuple[RepairPolicy, ...] = (
    RepairPolicy(code="missing_quantitative_results", target="quant", priority=100),
    RepairPolicy(code="missing_sentiment_analysis", target="psychologist", priority=90),
    RepairPolicy(code="missing_structured_data", target="retrieval", priority=80),
    RepairPolicy(code="low_retrieval_confidence", target="retrieval", priority=70),
    RepairPolicy(code="missing_risk_disclaimer", target="retrieval", priority=60),
    RepairPolicy(code="missing_data_provenance", target="retrieval", priority=50),
)


def resolve_repair_target(violations: list[Violation]) -> str:
    """Pick the highest-priority repair target for the given violations."""
    if not violations:
        return "retrieval"

    matched: list[RepairPolicy] = []
    codes = {violation.code for violation in violations}
    for policy in REPAIR_POLICIES:
        if policy.code in codes:
            matched.append(policy)

    if not matched:
        return "retrieval"

    best = max(matched, key=lambda policy: (policy.priority, policy.target))
    return best.target


def resolve_repair_target_from_codes(codes: list[str]) -> str:
    violations = [Violation(code=code, severity="high", message=code) for code in codes]
    return resolve_repair_target(violations)
