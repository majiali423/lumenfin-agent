from __future__ import annotations

import sys
from pathlib import Path

MCP_LAYER = Path(__file__).resolve().parents[1]
if str(MCP_LAYER) not in sys.path:
    sys.path.insert(0, str(MCP_LAYER))

import _bootstrap  # noqa: F401

from mcp.server.fastmcp import FastMCP

from adapters.doc_search import search_research_documents

mcp = FastMCP(
    "document-search",
    instructions=(
        "Search local research notes and return citable snippets with doc paths. "
        "Use for qualitative risk or supply-chain questions."
    ),
)


@mcp.tool()
def search_documents_tool(query: str, top_k: int = 3, company: str | None = None) -> dict:
    """Search research notes (Milvus hybrid when available, else keyword).

    Args:
        query: Natural language or keyword query.
        top_k: Maximum snippets to return.
        company: Optional company hint, e.g. Apple or Microsoft.
    """
    return search_research_documents(query, top_k=top_k, company=company)


if __name__ == "__main__":
    mcp.run()
