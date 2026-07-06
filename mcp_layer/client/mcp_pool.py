from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MCP_LAYER = Path(__file__).resolve().parents[1]
REPO_ROOT = MCP_LAYER.parent
PYTHON = sys.executable

SERVER_SCRIPTS = {
    "safe-calc": MCP_LAYER / "servers" / "safe_calc_server.py",
    "finance-db": MCP_LAYER / "servers" / "finance_db_server.py",
    "document-search": MCP_LAYER / "servers" / "document_server.py",
}


@dataclass
class ToolCallRecord:
    server: str
    tool: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class McpToolPool:
    """Connect to LumenFin MCP servers via stdio and expose unified tool calls."""

    audit: list[ToolCallRecord] = field(default_factory=list)
    _sessions: dict[str, ClientSession] = field(default_factory=dict)
    _stack: AsyncExitStack | None = None
    tool_index: dict[str, str] = field(default_factory=dict)

    async def connect(self) -> None:
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        for server_name, script_path in SERVER_SCRIPTS.items():
            params = StdioServerParameters(
                command=PYTHON,
                args=[str(script_path)],
                cwd=str(REPO_ROOT),
                env=os.environ.copy(),
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self._sessions[server_name] = session
            listed = await session.list_tools()
            for tool in listed.tools:
                self.tool_index[tool.name] = server_name

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None

    async def list_tools(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for server_name, session in self._sessions.items():
            tools = await session.list_tools()
            for tool in tools.tools:
                rows.append(
                    {
                        "name": tool.name,
                        "server": server_name,
                        "description": tool.description or "",
                    }
                )
        return rows

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        server_name = self.tool_index.get(tool_name)
        if not server_name:
            raise KeyError(f"Unknown tool: {tool_name}")
        session = self._sessions[server_name]
        result = await session.call_tool(tool_name, arguments)
        payload = _materialize_content(result.content)
        self.audit.append(
            ToolCallRecord(server=server_name, tool=tool_name, arguments=arguments, result=payload)
        )
        return payload


def _materialize_content(content: Any) -> Any:
    if not content:
        return None
    chunks: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(text)
    joined = "".join(chunks)
    try:
        return json.loads(joined)
    except json.JSONDecodeError:
        return joined


async def with_pool(coro):
    pool = McpToolPool()
    await pool.connect()
    try:
        return await coro(pool)
    finally:
        await pool.close()
