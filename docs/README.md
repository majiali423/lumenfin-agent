# LumenFin Documentation

## Start here

| Doc | Purpose |
|-----|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Current system spine |
| [CONFIGURATION.md](CONFIGURATION.md) | Environment variables and fail-closed defaults |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Install and validation commands |
| [VALIDATION_COMMANDS.md](VALIDATION_COMMANDS.md) | Supported offline / live / mutation commands |
| [PRODUCTION_LIMITATIONS.md](PRODUCTION_LIMITATIONS.md) | Portfolio RC boundary + validated gate summary |
| [PRODUCTION_BACKUP_RESTORE.md](PRODUCTION_BACKUP_RESTORE.md) | Backup, verification, restore rehearsal, and rollback |
| [MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md) | Tenant isolation scope and gaps |
| [PORTFOLIO_RELEASE_REPORT.md](PORTFOLIO_RELEASE_REPORT.md) | Release freeze evidence |
| [QUEUE_WORKER_INTEGRATION.md](QUEUE_WORKER_INTEGRATION.md) | Multi-process queue/worker evidence |
| [PROVIDER_RESILIENCE.md](PROVIDER_RESILIENCE.md) | Provider fault-injection evidence |
| [DEMO_GUIDE.md](DEMO_GUIDE.md) | Offline portfolio demo and optional live A/B/C walkthrough |
| [../LICENSE](../LICENSE) | MIT license for LumenFin-owned source |
| [../THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) | Dependency, image, provider, and data terms |

## Engineering evidence

| Doc | Purpose |
|-----|---------|
| [ENGINEERING_EVOLUTION.md](ENGINEERING_EVOLUTION.md) | Failure-driven evolution |
| [FINAL_RESULTS.md](FINAL_RESULTS.md) | Before → After summary |
| [FINAGENTBENCH_DESIGN.md](FINAGENTBENCH_DESIGN.md) | Evaluation design notes |
| [ARCHITECTURE_INDEX.md](ARCHITECTURE_INDEX.md) | Full map of current docs |

Optional local portfolio notes may live under `docs/portfolio/` and are not a
release gate.

## Subsystems

| Doc | Purpose |
|-----|---------|
| [architecture_decisions.md](architecture_decisions.md) | Durable design decisions |
| [RAG_MILVUS.md](RAG_MILVUS.md) | Hybrid RAG / Milvus Lite (dev) and Milvus Server (multi-process) |
| [MILVUS3_CUTOVER.md](MILVUS3_CUTOVER.md) | Milvus 3.0 volume/cutover contract |
| [BM25_CUTOVER.md](BM25_CUTOVER.md) | Native BM25 collection and rollback contract |
| [QWEN3_RERANK.md](QWEN3_RERANK.md) | Controlled Qwen3 rerank rollout, data-egress boundary, evaluation, and rollback |
| [TICKER_RESOLVE.md](TICKER_RESOLVE.md) | Ticker / company resolution |
| [HITL_CLARIFICATION.md](HITL_CLARIFICATION.md) | Clarification pause / resume |
| [INPUT_GUARDRAIL.md](INPUT_GUARDRAIL.md) | Input and upload guards |
| [MCP.md](MCP.md) | Optional MCP boundary |
| [evaluation_strategy.md](evaluation_strategy.md) | Internal evaluation strategy |
| [FINANCEBENCH_EVAL.md](FINANCEBENCH_EVAL.md) | External FinanceBench RAG retrieval eval (test-100 exploratory; confirmation-50 recorded and consumed; not product accuracy) |
| [ENCODING.md](ENCODING.md) | Windows / UTF-8 notes |
| [README_zh.md](README_zh.md) | Legacy Chinese stub — prefer root [../README.zh-CN.md](../README.zh-CN.md) |

## Reports

- Current: [`../reports/current/`](../reports/current/)
- Index: [`../reports/README.md`](../reports/README.md)

Older one-off reports were removed; recover from Git if needed
([`../reports/README.md`](../reports/README.md)).
