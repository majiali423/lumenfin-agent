"""Lightweight MCP agent client (no LangGraph) using DeepSeek + MCP tools."""
from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .mcp_pool import McpToolPool

SYSTEM_PROMPT = """You are a financial research assistant with access to MCP tools.

Rules:
1. For numeric answers, call query_company_metrics_tool then compute_ratio_tool — never mental math.
2. For qualitative risk questions, call search_documents_tool and cite doc paths from hits.
3. If a tool returns found=false or empty hits, say what is missing — do not invent data.
4. Respond in Chinese unless the user writes in English.

When you need a tool, reply with ONLY a JSON object (no markdown fences):
{"tool": "<tool_name>", "arguments": {<args>}}

When you have enough evidence for the final answer, reply with ONLY:
{"final": "<your answer with citations>"}
"""


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    if not api_key:
        raise RuntimeError("Set DEEPSEEK_API_KEY or OPENAI_API_KEY in environment / .env")
    return ChatOpenAI(model=model, api_key=api_key, base_url=base_url, temperature=0.1)


def _parse_agent_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    return json.loads(text)


async def run_agent_query(pool: McpToolPool, query: str, max_steps: int = 6) -> dict[str, Any]:
    llm = build_llm()
    tool_help = await pool.list_tools_for_prompt()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tool_help),
        HumanMessage(content=query),
    ]
    steps: list[dict[str, Any]] = []

    for _ in range(max_steps):
        response = await llm.ainvoke(messages)
        content = str(response.content)
        try:
            payload = _parse_agent_json(content)
        except json.JSONDecodeError:
            return {
                "answer": content,
                "steps": steps,
                "audit": [record.__dict__ for record in pool.audit],
                "note": "Model returned free text instead of JSON protocol",
            }

        if "final" in payload:
            return {
                "answer": payload["final"],
                "steps": steps,
                "audit": [record.__dict__ for record in pool.audit],
            }

        tool_name = payload.get("tool")
        arguments = payload.get("arguments") or {}
        if not tool_name:
            return {
                "answer": content,
                "steps": steps,
                "audit": [record.__dict__ for record in pool.audit],
                "error": "Invalid agent payload",
            }

        tool_result = await pool.call_tool(tool_name, arguments)
        steps.append({"tool": tool_name, "arguments": arguments, "result": tool_result})
        messages.append(response)
        messages.append(
            HumanMessage(content=f"Tool result for {tool_name}:\n{json.dumps(tool_result, ensure_ascii=False)}")
        )

    return {
        "answer": "Reached max tool steps without final answer.",
        "steps": steps,
        "audit": [record.__dict__ for record in pool.audit],
        "error": "max_steps",
    }
