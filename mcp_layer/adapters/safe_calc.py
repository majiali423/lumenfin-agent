"""Delegates to lumenfin.tools.safe_execute_formula (single source of truth)."""
from __future__ import annotations

from typing import Any

from lumenfin.tools import safe_execute_formula


def compute_ratio(formula: str, variables: dict[str, float]) -> dict[str, Any]:
    value = safe_execute_formula(formula.strip(), variables)
    return {
        "formula": formula,
        "variables": variables,
        "result": value,
        "engine": "lumenfin.tools.safe_execute_formula",
    }
