# MCP Tool Layer

LumenFin separates **workflow orchestration** (LangGraph) from **reusable tools** (MCP).

## Problem

Embedding every capability as a Python import ties tools to one runtime. MCP exposes the same financial primitives to:

- LumenFin LangGraph nodes (optional `MAS_TOOL_BACKEND=mcp`)
- `scripts/run_mcp_tools_demo.py` (no workflow)
- Cursor / other MCP clients

## Architecture

```text
MCP clients                LumenFin LangGraph (optional)
     |                              |
     | stdio MCP                    | MAS_TOOL_BACKEND=mcp
     v                              v
mcp_layer/servers/  ──adapters──>  src/lumenfin/
  safe-calc                         tools.safe_execute_formula
  finance-db                        sample_financial_data
  document-search                   (shared logic)
```

## Single source of truth

| MCP adapter | Core module |
|-------------|-------------|
| `mcp_layer/adapters/safe_calc.py` | `lumenfin.tools.safe_execute_formula` |
| `mcp_layer/adapters/finance_db.py` | `lumenfin.data.sample_financial_data` |

MCP servers are thin protocol wrappers. Changing quant logic in `lumenfin.tools` updates both the pipeline and MCP clients.

## What we do not claim

- Full LangGraph pipeline is not required to use the tools
- Not every node is MCP-backed by default (`MAS_TOOL_BACKEND=local` in tests)
- `document-search` uses bundled markdown notes; PDF Milvus RAG remains in the main workflow

## Commands

```powershell
python scripts/run_mcp_tools_demo.py
python scripts/run_mcp_tools_demo.py --json
python scripts/run_mcp_agent_demo.py
```

Environment:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MAS_TOOL_BACKEND` | `local` | Quant node uses in-process or MCP stdio |
| `MAS_MCP_DOC_SEARCH` | `auto` | `milvus` / `keyword` / `auto` for document-search server |

See also `mcp_layer/README.md` and `mcp_layer/cursor-mcp.example.json`.
