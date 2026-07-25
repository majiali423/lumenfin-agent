# MCP Tool Layer (optional side channel)

> **Production default:** keep `MAS_TOOL_BACKEND=local`. Diligence does **not** require MCP.
> **Hard boundary:** main evidence RAG runs only inside LangGraph **Retrieval** (`DocumentIndexer` + hybrid RRF).
> MCP `document-search` is an independent tool demo and **does not** write `rag_evidence` / diligence state.

LumenFin separates **workflow orchestration** (LangGraph) from **reusable tools** (MCP) so the same AST calculator (and demo data helpers) can be called from Cursor or scripts without importing the full graph.

## Production vs optional

| Path | Role | Deploy? |
|------|------|---------|
| LangGraph Retrieval + `DocumentIndexer` | **Production evidence RAG** (upload PDF → Milvus → `rag_evidence`) | Yes |
| SEC / Yahoo / upload metrics | **Production structured fundamentals** | Yes |
| MCP `safe-calc` | Optional: same AST via stdio (demo / Cursor) | Optional |
| MCP `finance-db` | Demo only: `SAMPLE_FINANCIAL_DATA` | Optional (never as live source) |
| MCP `document-search` | Demo sidecar: `mcp_layer/data/docs` notes (± ephemeral hybrid) | Optional |

`docker-compose.yml` and `scripts/run_live_showcase.py` **do not** start MCP servers. That is intentional: MCP is not the deploy baseline.

## What `MAS_TOOL_BACKEND=mcp` actually covers

Only **quant formula evaluation** (`safe_execute_formula` → MCP `compute_ratio_tool`).

It does **not** move Retrieval, RAG indexing, SEC, Yahoo, Critic, or HITL onto MCP. Saying “the whole graph runs on MCP” would be overclaiming.

## Architecture

```text
                    ┌─────────────────────────────────────┐
 Production         │ LangGraph: guardrail → … → Retrieval│
 diligence          │   DocumentIndexer / hybrid RAG      │
                    │   SEC/Yahoo → quant → critic → synth│
                    └─────────────────────────────────────┘

 Optional MCP       MCP clients (Cursor / scripts)
 (side channel)          |
                         | stdio
                         v
                    mcp_layer/servers/
                      safe-calc      → lumenfin.tools.safe_execute_formula
                      finance-db     → SAMPLE_FINANCIAL_DATA only
                      document-search→ mcp_layer/data/docs (± hybrid notes)
```

Adapters stamp every response with `production_scope` (see `mcp_layer/adapters/scope.py`):

- `affects_diligence_state: false`
- `data_contract`: `ast_formula_only` | `sample_financial_data_only` | `mcp_research_notes_sidecar`

## Single source of truth (logic, not product path)

| MCP adapter | Core module | Product meaning |
|-------------|-------------|-----------------|
| `adapters/safe_calc.py` | `lumenfin.tools.safe_execute_formula` | Same math as quant |
| `adapters/finance_db.py` | `lumenfin.data.sample_financial_data` | **Sample only** ≠ live fundamentals |
| `adapters/doc_search.py` | notes under `mcp_layer/data/docs` (+ optional Milvus over those notes) | **Not** upload-PDF production RAG |

## document-search modes (sidecar only)

| `MAS_MCP_DOC_SEARCH` | Behavior |
|----------------------|----------|
| `keyword` | Keyword over bundled markdown/txt notes |
| `milvus` / `hybrid` | Ephemeral hybrid over **those same notes** (not user upload index) |
| `auto` (default) | Try hybrid, fall back to keyword |

Even in `milvus` mode, hits are **not** attached to diligence `rag_evidence` unless a custom client copies them. For production PDF RAG see [`RAG_MILVUS.md`](RAG_MILVUS.md).

## Commands (demo)

```powershell
python scripts/run_mcp_tools_demo.py
python scripts/run_mcp_tools_demo.py --json
python scripts/run_mcp_agent_demo.py
```

## Environment

| Variable | Production recommendation | Meaning |
|----------|---------------------------|---------|
| `MAS_TOOL_BACKEND` | **`local`** | `mcp` only if you explicitly want stdio for ratio demos |
| `MAS_MCP_DOC_SEARCH` | `keyword` or leave unset for demos | Sidecar search mode; irrelevant to production RAG |

See also `mcp_layer/README.md` and `mcp_layer/cursor-mcp.example.json`.
