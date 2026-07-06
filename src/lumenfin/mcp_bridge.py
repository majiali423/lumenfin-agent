"""Sync bridge from LangGraph nodes to MCP servers (stdio protocol)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MCP_LAYER = _REPO_ROOT / "mcp_layer"


def _ensure_mcp_layer_on_path() -> None:
    path = str(_MCP_LAYER)
    if path not in sys.path:
        sys.path.insert(0, path)


def call_mcp_tool_sync(tool_name: str, arguments: dict[str, Any]) -> Any:
    _ensure_mcp_layer_on_path()
    from client.mcp_pool import with_pool

    async def _run(pool) -> Any:
        return await pool.call_tool(tool_name, arguments)

    return asyncio.run(with_pool(_run))


def compute_ratio_via_mcp(formula: str, variables: dict[str, float]) -> float:
    payload = call_mcp_tool_sync(
        "compute_ratio_tool",
        {"formula": formula, "variables": variables},
    )
    if isinstance(payload, dict) and payload.get("result") is not None:
        return float(payload["result"])
    raise ValueError(f"Unexpected MCP payload: {payload}")
