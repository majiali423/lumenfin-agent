"""Run MCP tool calls without LangGraph — proves tools are reusable outside LumenFin."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp_layer"))

from client.mcp_pool import McpToolPool, with_pool


async def _demo(pool: McpToolPool) -> dict:
    tools = await pool.list_tools()
    metrics = await pool.call_tool(
        "query_company_metrics_tool",
        {"company": "Apple", "metrics": ["revenue_2025", "r_and_d_2025"]},
    )
    ratio = await pool.call_tool(
        "compute_ratio_tool",
        {
            "formula": "r_and_d_2025 / revenue_2025",
            "variables": {
                "r_and_d_2025": metrics["metrics"]["r_and_d_2025"]["value"],
                "revenue_2025": metrics["metrics"]["revenue_2025"]["value"],
            },
        },
    )
    docs = await pool.call_tool(
        "search_documents_tool",
        {"query": "supply chain risk", "top_k": 2},
    )
    return {
        "tools": tools,
        "metrics": metrics,
        "ratio": ratio,
        "documents": docs,
        "audit": [record.__dict__ for record in pool.audit],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LumenFin MCP tool demo (no LangGraph)")
    parser.add_argument("--json", action="store_true", help="Print JSON output")
    args = parser.parse_args()

    result = asyncio.run(with_pool(_demo))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print("Registered MCP tools:")
    for tool in result["tools"]:
        print(f"  - {tool['name']} [{tool['server']}]")
    print("\nApple R&D intensity:", result["ratio"]["result"])
    print("Document hits:", len(result["documents"].get("hits", [])))
    print("\nThis demo runs without the LangGraph pipeline — any MCP client can call the same servers.")


if __name__ == "__main__":
    main()
