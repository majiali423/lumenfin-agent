"""Delegates to lumenfin.tools.safe_execute_formula (single source of truth)."""
from __future__ import annotations

from typing import Any

from lumenfin.tools import safe_execute_formula

from .scope import SCOPE_SAFE_CALC, stamp_scope


def compute_ratio(formula: str, variables: dict[str, float]) -> dict[str, Any]:
    value = safe_execute_formula(formula.strip(), variables)
    return stamp_scope(
        {
            "formula": formula,
            "variables": variables,
            "result": value,
            "engine": "lumenfin.tools.safe_execute_formula",
        },
        SCOPE_SAFE_CALC,
    )
