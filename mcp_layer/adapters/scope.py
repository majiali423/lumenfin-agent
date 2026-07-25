"""Explicit production-scope tags for MCP tool responses.

MCP is an optional side channel for demos / Cursor. It is not the diligence
pipeline's evidence or fundamentals path. Callers (and docs) should treat
these tags as the contract.
"""
from __future__ import annotations

from typing import Any

# Hard boundary: MCP tools do not write diligence state / rag_evidence.
PRODUCTION_ROLE = "optional_side_channel"
NOT_DILIGENCE_STATE = (
    "MCP results do not enter LangGraph diligence state "
    "(retrieved_docs / rag_evidence / final_report) unless a custom client copies them."
)

SCOPE_SAFE_CALC = {
    "production_role": PRODUCTION_ROLE,
    "data_contract": "ast_formula_only",
    "affects_diligence_state": False,
    "note": (
        "Same AST engine as quant. MAS_TOOL_BACKEND=mcp only routes formula evaluation "
        "through MCP stdio; retrieval/RAG/SEC stay in-process."
    ),
}

SCOPE_FINANCE_DB = {
    "production_role": PRODUCTION_ROLE,
    "data_contract": "sample_financial_data_only",
    "affects_diligence_state": False,
    "note": (
        "Reads SAMPLE_FINANCIAL_DATA only. Not SEC/Yahoo/upload fundamentals. "
        "Do not treat as live-mode structured_source."
    ),
}

SCOPE_DOCUMENT_SEARCH = {
    "production_role": PRODUCTION_ROLE,
    "data_contract": "mcp_research_notes_sidecar",
    "affects_diligence_state": False,
    "note": (
        "Searches mcp_layer/data/docs (and optional ephemeral hybrid over those notes). "
        "Production PDF evidence RAG lives in LangGraph Retrieval + DocumentIndexer; "
        "see docs/RAG_MILVUS.md."
    ),
}


def stamp_scope(payload: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with production-scope metadata attached."""
    out = dict(payload)
    out["production_scope"] = dict(scope)
    out.setdefault("affects_diligence_state", scope.get("affects_diligence_state", False))
    return out
