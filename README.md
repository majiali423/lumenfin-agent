# LumenFin Agent

> Evidence-grounded multi-agent financial diligence workbench (not a chat wrapper).

Traceable LangGraph workflow | Hybrid Milvus RAG | AST-safe metrics | Offline test harness

[Chinese overview](docs/README_zh.md) | [Quick Start](#quick-start) | [Architecture decisions](docs/architecture_decisions.md)

---

## What it is

LumenFin turns **user queries and uploaded financial PDFs** into a **structured diligence report** through an explicit LangGraph pipeline:

- Each step is logged to `audit_log` and exported as JSON artifacts.
- Numeric ratios are computed by an **AST-safe expression engine**, not by the LLM.
- PDF evidence is retrieved via **Milvus Lite hybrid RAG** (vector + keyword + RRF) with page citations.
- Missing data triggers **replanner / degraded mode** instead of silent hallucination.

LumenFin can also export its run state into the neutral `FinRun` schema used by
FinAgentBench:

```powershell
python run_demo.py --query "Compare Apple and Microsoft FY2025 financial performance, supply chain risk, and market data quality." --thread-id lumenfin-e2e --output-dir outputs
$state = Get-ChildItem outputs\lumenfin-e2e_*_state.json | Sort-Object LastWriteTime -Descending | Select-Object -First 1
python scripts\export_finrun.py $state.FullName --out outputs\lumenfin-e2e-finrun.json
```

The benchmark project can then evaluate either the raw LumenFin `*_state.json`
with `--adapter lumenfin` or the exported `FinRun` JSON. This keeps LumenFin as
the generation/runtime project and FinAgentBench as the downstream reliability
gate.

---

## Positioning (vs. a RAG chatbot)

| Dimension | Typical RAG chatbot | LumenFin |
|-----------|---------------------|--------------|
| Orchestration | Single prompt / ReAct loop | LangGraph explicit state machine |
| Numbers | Model narration | `quant` node AST evaluation |
| Evidence | Optional citations | Hybrid RAG + `rag_evidence` + page cites |
| Quality | Subjective reading | Golden eval / RAG metrics / trace scorer |
| Failure | Hallucinate or crash | Replanner -> degraded mode |

---

## Architecture

```mermaid
flowchart TB
  subgraph input [Input]
    Q[User Query]
    PDF[Uploaded PDF]
  end

  subgraph graph [LangGraph Pipeline]
    IG[Input Guardrail]
    QP[Query Planner]
    SV[Supervisor]
    RT[Retrieval]
    QN[Quantitative Analyst]
    PS[Psychologist]
    CR[Critic]
    RP2[Repair router-retry]
    RP[Replanner]
    SY[Synthesizer]
  end

  subgraph output [Artifacts]
    RPT[report.md]
    AUD[audit.json]
    ST[state.json]
    MAN[manifest.json]
  end

  Q --> IG
  PDF --> IG
  IG --> QP --> SV --> RT --> QN --> PS --> CR
  CR -->|pass| SY
  CR -->|needs fix| RP2
  RP2 --> RT
  RP2 --> QN
  RP2 --> PS
  CR -.->|max iterations| SY
  RT -.->|data gap| RP
  QN -.->|data gap| RP
  RP -.->|retry| RT
  RP -.->|give up| SY
  SY --> RPT
  SY --> AUD
  SY --> ST
  SY --> MAN
```

**Repair note:** `repair` is an **evaluator-router-retry prototype** (routes back to retrieval/quant/sentiment based on critic findings). It is not yet a full LLM-based repair policy that rewrites queries or patches data.

**Retrieval detail:** PDF -> page chunks -> Milvus Lite -> vector + keyword -> RRF -> `filename#p{page}` citations.

---

## Quick Start

### 0. Encoding (Windows)

PowerShell on Windows may use a legacy code page; run UTF-8 setup before running the app:

```powershell
cd lumenfin-agent
. .\scripts\ensure_utf8.ps1
```

Or use VS Code task: **Tasks -> Run Task -> Run Demo (UTF-8)**.

All repository text files are UTF-8 (see `.gitattributes`, `.editorconfig`, [docs/ENCODING.md](docs/ENCODING.md)).

### 1. Install (once)

```powershell
cd lumenfin-agent
.\.venv\Scripts\pip install -e .
.\.venv\Scripts\python -c "import lumenfin; print('OK')"
```

### 2. Configure LLM

```powershell
copy .env.example .env
# Edit .env - set DEEPSEEK_API_KEY=sk-...
```

In `APP_ENV=dev|test`, running without a DeepSeek key uses `local-fallback`
(good for wiring tests; weaker reports). In non-dev environments, local fallback
is disabled unless `ALLOW_LOCAL_FALLBACK=true` is set explicitly.

### 3. CLI demo

```powershell
. .\scripts\ensure_utf8.ps1
.\.venv\Scripts\python run_demo.py --thread-id learning-001
```

Outputs under `outputs/`:

```text
learning-001_*_report.md
learning-001_*_audit.json
learning-001_*_state.json
learning-001_*_manifest.json
```

Deterministic offline demo (no external LLM or market-data API):

```powershell
.\.venv\Scripts\python scripts\run_portfolio_demo.py --write
```

This prints a compact JSON summary with workflow status, companies, evaluator score, audit steps, and exported artifact paths.

### 4. Web UI + PDF upload

```powershell
. .\scripts\ensure_utf8.ps1
.\.venv\Scripts\python start_api.py
```

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000 | Web UI |
| http://127.0.0.1:8000/docs | OpenAPI |

HITL: ambiguous queries may return `workflow_status: needs_clarification`; resume with `POST /api/v1/clarify`. State is persisted in **SQLite** (`workflow_checkpoints`) so API restart can resume. See [docs/HITL_CLARIFICATION.md](docs/HITL_CLARIFICATION.md).

Structured metrics (no PDF): `POST /api/v1/analyze-data` with a `company_metrics` object keyed by company name.

```json
{
  "query": "Compare FY2025 margins",
  "company_metrics": {
    "NVIDIA": { "revenue_2025": 130.5, "ebitda_2025": 75.2 }
  }
}
```

### 5. Tests and eval

Offline tests (mock LLM + fake market data, no external API):

```powershell
.\.venv\Scripts\python scripts\run_tests.py
```

Optional live integration test (requires DeepSeek key in `.env`):

```powershell
$env:RUN_INTEGRATION_TESTS = "1"
.\.venv\Scripts\python scripts\run_tests.py --integration
```

Eval scripts:

```powershell
.\.venv\Scripts\python scripts\run_golden_eval.py --write
.\.venv\Scripts\python scripts\run_rag_eval.py
```

`run_golden_eval.py` is intended as a live-quality regression check. It may call the configured LLM and market-data provider; use `run_portfolio_demo.py --write` for a deterministic offline portfolio demo.

---

## Capabilities

| Area | What is implemented |
|------|---------------------|
| Orchestration | LangGraph `StateGraph`, conditional edges, SQLite HITL checkpoint |
| Hybrid RAG | Milvus Lite + keyword + RRF; page-level citations in state |
| Deterministic quant | AST-safe formulas for margins, intensity, derived ratios |
| Offline tests | 85+ unit tests; default harness avoids DeepSeek/Yahoo |
| RAG metrics | Recall@K, MRR, citation coverage via `run_rag_eval.py` |
| HITL | Clarification pause + `/clarify` resume; **SQLite-backed** checkpoint |
| Run manifest | `*_manifest.json` with latency, tokens, evaluator, `data_sources` |
| Offline portfolio demo | `scripts/run_portfolio_demo.py --write` produces a deterministic offline report + eval artifacts |
| Input guardrail | PDF injection pattern scan (EN + Unicode CJK patterns) |
| Parallel fan-out | Per-company thread pool in retrieval / quant / sentiment |
| Telemetry | `audit_log` latency/tokens on **all pipeline nodes** |
| Repair loop | Critic-driven router-retry prototype (max 2 iterations) |
| Structured ingest | JSON metrics API (`/api/v1/analyze-data`) + JSON/CSV/Excel/Markdown file upload |
| Tool transport | `local` in-process or `mcp` stdio for quant ratios |
| MCP tool layer | `mcp_layer/servers` + `scripts/run_mcp_tools_demo.py` |

### Non-goals (v0.3)

- Production multi-tenant auth / RBAC (local demos may leave `MAS_API_KEY` empty when `APP_ENV=dev|test`; non-dev requires a key)
- Production defaults to `DATA_MODE=live` and disables local LLM fallback unless explicitly overridden for a demo.
- Full LangGraph Postgres channel saver (snapshot checkpoint only; compose Postgres is optional infra, not a completed multi-tenant store)
- Shared Milvus Lite across API + CLI + worker processes (Lite uses a single-writer file lock; production needs a real Milvus service)
- Image/chart OCR upload (use PDF or structured files)
- Investment advice or trade execution
- Silent sample fundamentals in `DATA_MODE=live` (demo mode keeps the sample DB and labels it explicitly)

See [docs/architecture_decisions.md](docs/architecture_decisions.md) for design rationale.

### MCP tool layer

Financial primitives are also exposed as MCP servers under `mcp_layer/` (reusable outside LangGraph).

```powershell
.\.venv\Scripts\python scripts\run_mcp_tools_demo.py
.\.venv\Scripts\python scripts\run_mcp_agent_demo.py
```

See [docs/MCP.md](docs/MCP.md).

---

## Project layout

```text
run_demo.py / start_api.py
src/lumenfin/
  graph.py          # LangGraph wiring
  agents.py         # node implementations
  rag/              # Milvus hybrid retrieval
  input_guardrail.py
scripts/run_tests.py
outputs/            # generated artifacts (gitignored)
```

---

## Further reading

- [Architecture decisions](docs/architecture_decisions.md)
- [Chinese overview](docs/README_zh.md)
- [Milvus RAG design](docs/RAG_MILVUS.md)
- [PDF input guardrail](docs/INPUT_GUARDRAIL.md)
- [HITL clarification](docs/HITL_CLARIFICATION.md)
- [Evaluation strategy](docs/evaluation_strategy.md)

---

## Tech stack

| Layer | Choice |
|-------|--------|
| Orchestration | LangGraph |
| LLM | DeepSeek + local fallback |
| Vector DB | Milvus Lite (no Docker) |
| PDF | PyMuPDF |
| API | FastAPI |
| Persistence | SQLite |

---

## License / disclaimer

AI-generated research output for demonstration only. Not investment advice.
