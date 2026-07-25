# LumenFin Architecture Index (Release Candidate / Final Packaging)

Generated for documentation organization. This index does **not** change runtime behavior.

## Final packaging (start here)

| Doc | Purpose |
|-----|---------|
| [FINAL_ARCHITECTURE.md](FINAL_ARCHITECTURE.md) | System overview, diagram, data flow |
| [ENGINEERING_EVOLUTION.md](ENGINEERING_EVOLUTION.md) | Phase 1–6 failure-driven history |
| [FINAGENTBENCH_DESIGN.md](FINAGENTBENCH_DESIGN.md) | Reliability evaluation framework design |
| [portfolio/INTERVIEW_NOTES.md](portfolio/INTERVIEW_NOTES.md) | Optional portfolio Q&A (not a release gate) |
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | Before → After + limitations |
| [CONFIGURATION.md](CONFIGURATION.md) | Runtime/provider configuration |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Locked install + release gates |
| [PRODUCTION_LIMITATIONS.md](PRODUCTION_LIMITATIONS.md) | Controlled deployment boundary |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Offline and live demo runbook |

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
| [MCP.md](MCP.md) | MCP tool layer boundary |
| [../mcp_layer/README.md](../mcp_layer/README.md) | MCP servers/adapters |

## Data & retrieval

| Doc | Purpose |
|-----|---------|
| [RAG_MILVUS.md](RAG_MILVUS.md) | Milvus Lite hybrid RAG |
| [RAG_PRODUCTION_PLAN.md](RAG_PRODUCTION_PLAN.md) | Production RAG roadmap (non-goals: RAG≠AST) |
| [TICKER_RESOLVE.md](TICKER_RESOLVE.md) | Ticker / company resolution |
| [TOOLS_HONESTY_FIXES.md](TOOLS_HONESTY_FIXES.md) | Tool honesty / provenance |

## Safety & control plane

| Doc | Purpose |
|-----|---------|
| [INPUT_GUARDRAIL.md](INPUT_GUARDRAIL.md) | Upload/query guardrails |
| [HITL_CLARIFICATION.md](HITL_CLARIFICATION.md) | Clarification pause / resume |
| [AGENT_NODE_OPTIMIZATIONS.md](AGENT_NODE_OPTIMIZATIONS.md) | Node cost/latency notes |

## Ops

| Doc | Purpose |
|-----|---------|
| [ENCODING.md](ENCODING.md) | Windows/UTF-8 notes |

## External evaluation (FinAgentBench)

Primary repo: sibling `../finagentbench-demo/docs/`

| Doc | Purpose |
|-----|---------|
| `architecture.md` | Bench architecture |
| `finrun_schema.md` | Canonical FinRun schema |
| `adapter_guide.md` / `agent_integration_guide.md` | Adapter contract |
| `lumenfin_case_selection.md` / `lumenfin_regression_case.md` | Case design |

## Release artifacts

| Report | Role |
|--------|------|
| [LumenFin_Final_Release_Report.md](../reports/current/LumenFin_Final_Release_Report.md) | LumenFin release evidence |
| [Joint_Compatibility_Report.md](../reports/current/Joint_Compatibility_Report.md) | Cross-repository contract |
| [LumenFin_RC_Final_Reliability_Report.md](../reports/current/LumenFin_RC_Final_Reliability_Report.md) | Live RC pack evidence |
| [Release_Checklist.md](../Release_Checklist.md) | Blocking release checklist |

## Non-goals (do not regress)

- Do not replace AST/SEC fundamentals with pure RAG
- Do not prompt-force citations
- Do not lower FinAgentBench thresholds to pass RC
- Do not invent numbers on `incomplete_data`
