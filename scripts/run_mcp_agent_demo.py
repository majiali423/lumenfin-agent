"""Run a minimal LangChain agent on top of LumenFin MCP servers (no LangGraph)."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp_layer"))

from client.agent import run_agent_query
from client.mcp_pool import McpToolPool, with_pool


def main() -> None:
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="LumenFin MCP agent demo (no LangGraph)")
    parser.add_argument(
        "--query",
        default="计算 Apple 2025 年研发投入占收入的比例，并引用文档说明供应链风险。",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    async def _run(pool: McpToolPool):
        return await run_agent_query(pool, args.query)

    result = asyncio.run(with_pool(_run))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(result["answer"])
    print("\n--- tool audit ---")
    for row in result.get("audit", []):
        print(f"[{row['server']}] {row['tool']}({json.dumps(row['arguments'], ensure_ascii=False)})")


if __name__ == "__main__":
    main()
