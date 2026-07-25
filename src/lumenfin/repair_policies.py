from __future__ import annotations

from .artifacts import RepairPolicy, Violation

# Only these critic codes justify a full retrieval re-run.
RETRIEVAL_WORTHY_CODES = frozenset(
    {
        "missing_structured_data",
        "low_retrieval_confidence",
    }
)

REPAIR_POLICIES: tuple[RepairPolicy, ...] = (
    RepairPolicy(code="missing_quantitative_results", target="quant", priority=100),
    RepairPolicy(code="missing_sentiment_analysis", target="psychologist", priority=90),
    RepairPolicy(code="missing_structured_data", target="retrieval", priority=80),
    RepairPolicy(code="low_retrieval_confidence", target="retrieval", priority=70),
    # Report-template gaps are synthesizer concerns; never route to retrieval.
    RepairPolicy(code="missing_risk_disclaimer", target="quant", priority=20),
    RepairPolicy(code="missing_data_provenance", target="quant", priority=10),
)

# Safe default when no policy matches — avoid blind retrieval fan-out.
_DEFAULT_REPAIR_TARGET = "quant"


def resolve_repair_target(violations: list[Violation]) -> str:
    """Pick the highest-priority repair target for the given violations."""
    if not violations:
        return _DEFAULT_REPAIR_TARGET

    matched: list[RepairPolicy] = []
    codes = {violation.code for violation in violations}
    for policy in REPAIR_POLICIES:
        if policy.code in codes:
            matched.append(policy)

    if not matched:
        return _DEFAULT_REPAIR_TARGET

    best = max(matched, key=lambda policy: (policy.priority, policy.target))
    target = best.target
    if target == "retrieval" and not codes.intersection(RETRIEVAL_WORTHY_CODES):
        return _DEFAULT_REPAIR_TARGET
    return target


def resolve_repair_target_from_codes(codes: list[str]) -> str:
    violations = [Violation(code=code, severity="high", message=code) for code in codes]
    return resolve_repair_target(violations)
