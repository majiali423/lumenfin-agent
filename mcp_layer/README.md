# LumenFin MCP Tool Layer

Financial capabilities exposed as **standalone MCP servers**. Any MCP client (Cursor, a lightweight agent, or future projects) can call the same tools without importing the LangGraph workflow.

## Why MCP here?

| In-process (`providers/`) | MCP (`mcp_layer/`) |
|---------------------------|---------------------|
| Fast path inside LumenFin API | Standard `list_tools` / `call_tool` protocol |
| Tight coupling to one runtime | Reusable across clients and processes |
| Good for tests / default mode | Good for demos of tool-layer decoupling |

Core logic is **not duplicated**: adapters call `lumenfin.tools`, `sample_financial_data`, etc.

## Servers

| Server | Tool | LumenFin equivalent |
|--------|------|---------------------|
| `safe-calc` | `compute_ratio_tool` | `quant` node AST engine |
| `finance-db` | `query_company_metrics_tool` | sample DB retrieval |
| `document-search` | `search_documents_tool` | qualitative note search |

## Quick demo (no LangGraph)

```powershell
cd lumenfin-agent
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\python scripts\run_mcp_tools_demo.py
```

LangChain agent demo (optional extra: `pip install -e ".[mcp-agent]"`, needs `DEEPSEEK_API_KEY`):

```powershell
.\.venv\Scripts\python scripts\run_mcp_agent_demo.py
```

`document-search` uses Milvus hybrid RAG when available (`MAS_MCP_DOC_SEARCH=auto`).

## Cursor integration

Copy `mcp_layer/cursor-mcp.example.json` into your Cursor MCP settings and set `cwd` to this repo root.

## Run a server manually

```powershell
python mcp_layer/servers/safe_calc_server.py
```
