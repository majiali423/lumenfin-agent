from __future__ import annotations


def classify_critic_repair_target(findings: list[str]) -> str:
    """Map compliance findings to the next node in the evaluator-optimizer loop."""
    joined = " ".join(findings).lower()
    if "missing quantitative" in joined or "quantitative results" in joined:
        return "quant"
    if "missing sentiment" in joined:
        return "psychologist"
    return "retrieval"
