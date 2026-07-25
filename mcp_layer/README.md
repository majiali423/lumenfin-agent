# LumenFin MCP Tool Layer

Standalone MCP servers for **optional demos** (Cursor, scripts, interview tool-decoupling narrative).

## Production rule (read this first)

| Do in production | Do **not** treat MCP as |
|------------------|-------------------------|
| LangGraph Retrieval + `DocumentIndexer` for PDF evidence | The diligence RAG path |
| SEC / Yahoo / upload metrics for numbers | Live fundamentals (`finance-db` is sample-only) |
| `MAS_TOOL_BACKEND=local` | “Whole graph over MCP” |

**Hard boundary:** main evidence RAG lives only in LangGraph Retrieval and writes `rag_evidence`.
MCP `document-search` is a sidecar over `mcp_layer/data/docs` and **does not** enter diligence state by default.

Full contract: [`docs/MCP.md`](../docs/MCP.md). Production RAG: [`docs/RAG_MILVUS.md`](../docs/RAG_MILVUS.md).

## Why MCP exists

| In-process (`MAS_TOOL_BACKEND=local`) | MCP (`mcp_layer/`) |
|---------------------------------------|---------------------|
| Production / CI default | Standard `list_tools` / `call_tool` for external clients |
| Fast, no stdio | Demo of tool-layer decoupling |
| Used by API + showcase | **Not** started by docker-compose or live showcase |

Core math is **not duplicated**: `safe-calc` calls `lumenfin.tools.safe_execute_formula`.

## Servers

| Server | Tool | Data contract | Diligence state? |
|--------|------|---------------|------------------|
| `safe-calc` | `compute_ratio_tool` | AST formula only | No (unless quant uses `MAS_TOOL_BACKEND=mcp` for the formula call) |
| `finance-db` | `query_company_metrics_tool` | `SAMPLE_FINANCIAL_DATA` only | No |
| `document-search` | `search_documents_tool` | Research notes sidecar (± ephemeral hybrid on those notes) | No |

Every adapter response includes `production_scope` + `affects_diligence_state: false`.

## document-search (sidecar)

`MAS_MCP_DOC_SEARCH=auto|keyword|milvus`:

- Corpus = `mcp_layer/data/docs` (markdown/txt), **not** user-upload PDF index from `/documents/index`.
- `milvus`/`auto` may build an **ephemeral** hybrid session over those notes; this is still not production upload RAG.

## Quick demo (no LangGraph)

```powershell
cd lumenfin-agent
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\python scripts\run_mcp_tools_demo.py
```

Optional agent demo (`pip install -e ".[mcp-agent]"`, needs `DEEPSEEK_API_KEY`):

```powershell
.\.venv\Scripts\python scripts\run_mcp_agent_demo.py
```

## Cursor integration

Copy `mcp_layer/cursor-mcp.example.json` into Cursor MCP settings and set `cwd` to this repo root.

## Run a server manually

```powershell
python mcp_layer/servers/safe_calc_server.py
```
