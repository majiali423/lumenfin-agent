from __future__ import annotations

import sys
from pathlib import Path

MCP_LAYER = Path(__file__).resolve().parents[1]
if str(MCP_LAYER) not in sys.path:
    sys.path.insert(0, str(MCP_LAYER))

import _bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

from adapters.safe_calc import compute_ratio

mcp = FastMCP(
    "safe-calc",
    instructions=(
        "Deterministic financial ratio calculator backed by LumenFin AST engine. "
        "Use when numeric precision matters; never guess arithmetic."
    ),
)


@mcp.tool()
def compute_ratio_tool(formula: str, variables: dict[str, float]) -> dict:
    """Evaluate a safe arithmetic formula.

    Args:
        formula: Expression like r_and_d_2025 / revenue_2025
        variables: Numeric bindings, e.g. r_and_d_2025=33.4, revenue_2025=412.0
    """
    return compute_ratio(formula, variables)


if __name__ == "__main__":
    mcp.run()
