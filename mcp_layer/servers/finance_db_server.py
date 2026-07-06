from __future__ import annotations

import sys
from pathlib import Path

MCP_LAYER = Path(__file__).resolve().parents[1]
if str(MCP_LAYER) not in sys.path:
    sys.path.insert(0, str(MCP_LAYER))

import _bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

from adapters.finance_db import query_company_metrics

mcp = FastMCP(
    "finance-db",
    instructions=(
        "Structured financial metrics from LumenFin sample_financial_data. "
        "Always cite the returned source field; do not invent numbers."
    ),
)


@mcp.tool()
def query_company_metrics_tool(company: str, metrics: list[str] | None = None) -> dict:
    """Fetch company metrics from the LumenFin sample dataset.

    Args:
        company: Company name, e.g. Apple, Microsoft, NVIDIA.
        metrics: Optional metric keys such as revenue_2025, r_and_d_2025.
    """
    return query_company_metrics(company, metrics)


if __name__ == "__main__":
    mcp.run()
