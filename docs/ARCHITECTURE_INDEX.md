# LumenFin Architecture Index

Generated for documentation organization. This index does **not** change runtime
behavior.

## Final packaging (start here)

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System overview, diagram, data flow |
| [ENGINEERING_EVOLUTION.md](ENGINEERING_EVOLUTION.md) | Phase 1–6 failure-driven history |
| [FINAGENTBENCH_DESIGN.md](FINAGENTBENCH_DESIGN.md) | Reliability evaluation framework design |
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | Before → After packaging snapshot (banner may mark historical rows) |
| [CONFIGURATION.md](CONFIGURATION.md) | Runtime/provider configuration |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Locked install + release gates |
| [PRODUCTION_LIMITATIONS.md](PRODUCTION_LIMITATIONS.md) | Controlled deployment boundary + validated gate summary |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Offline and optional live demo runbook |
| [PORTFOLIO_RELEASE_REPORT.md](PORTFOLIO_RELEASE_REPORT.md) | Current RC freeze evidence |

Local interview notes under `docs/portfolio/` are gitignored and are **not**
part of the public release surface.

## System spine

```text
query → input_guardrail → query_planner → (HITL?) → supervisor
     → retrieval (RAG + issuer SEC Financial Grounding)
     → quant → psychologist → critic → (repair?)
     → claim_binder → synthesizer
     → export_finrun_state() → FinAgentBench
```

Reliability layers (current):

1. **Issuer isolation** — document entity resolution; supervisor expands issuers only
2. **Financial Grounding** — issuer-only SEC/Yahoo gap-fill when uploads are not AST-computable
3. **Claim → Evidence Binding** — verified claims only enter material report assertions
4. **Fail-closed** — `incomplete_data` when fundamentals absent; no invented AST ratios

## Core architecture

| Doc | Purpose |
|-----|---------|
| [architecture_decisions.md](architecture_decisions.md) | Why LangGraph, planner, AST quant, fail-loud |
| [evaluation_strategy.md](evaluation_strategy.md) | How LumenFin evaluates diligence quality |
| [FINANCEBENCH_EVAL.md](FINANCEBENCH_EVAL.md) | External FinanceBench retrieval eval harness |
| [FINANCEBENCH_NEXT_PHASE.md](FINANCEBENCH_NEXT_PHASE.md) | LEDGER public-dev sealed/stopped; further ranking work needs a new unseen holdout |
| [MCP.md](MCP.md) | MCP tool layer boundary |
| [../mcp_layer/README.md](../mcp_layer/README.md) | MCP servers/adapters |

## Data & retrieval

| Doc | Purpose |
|-----|---------|
| [RAG_MILVUS.md](RAG_MILVUS.md) | Hybrid RAG / Milvus Lite + Server |
| [BM25_CUTOVER.md](BM25_CUTOVER.md) | Native BM25 v4 cutover |
| [MILVUS3_CUTOVER.md](MILVUS3_CUTOVER.md) | Milvus 3.0 cutover |
| [QWEN3_RERANK.md](QWEN3_RERANK.md) | Qwen3 rerank rollout |
| [TICKER_RESOLVE.md](TICKER_RESOLVE.md) | Ticker / company resolution |
| [QUEUE_WORKER_INTEGRATION.md](QUEUE_WORKER_INTEGRATION.md) | Multi-process queue/worker evidence |
| [PROVIDER_RESILIENCE.md](PROVIDER_RESILIENCE.md) | Provider fault-injection evidence |

## Safety & control plane

| Doc | Purpose |
|-----|---------|
| [INPUT_GUARDRAIL.md](INPUT_GUARDRAIL.md) | Upload/query guardrails |
| [HITL_CLARIFICATION.md](HITL_CLARIFICATION.md) | Clarification pause / resume |

## Ops

| Doc | Purpose |
|-----|---------|
| [ENCODING.md](ENCODING.md) | Windows/UTF-8 notes |
| [PRODUCTION_BACKUP_RESTORE.md](PRODUCTION_BACKUP_RESTORE.md) | Backup / restore rehearsal |

## External evaluation (FinAgentBench)

Primary repo: sibling `../finagentbench-demo/docs/`

| Doc | Purpose |
|-----|---------|
| `architecture.md` | Bench architecture |
| `finrun_schema.md` | Canonical FinRun schema |
| `adapter_guide.md` / `agent_integration_guide.md` | Adapter contract |
| `lumenfin_case_selection.md` / `lumenfin_regression_case.md` | Case design |

## Release artifacts

| Doc | Purpose |
|-----|---------|
| [PORTFOLIO_RELEASE_REPORT.md](PORTFOLIO_RELEASE_REPORT.md) | Current freeze |
| [../Release_Checklist.md](../Release_Checklist.md) | Historical rc1 checklist (bannered) |
| [../reports/current/](../reports/current/) | Current evidence only |
| [../reports/README.md](../reports/README.md) | Pointer: recover old snapshots from Git |
