# LumenFin documentation

## Start with the product

| Document | Purpose |
|---|---|
| [Project overview](../README.md) · [中文](../README.zh-CN.md) | Product scenarios, local setup and version status |
| [Demo guide](DEMO_GUIDE.md) | Offline walkthrough and optional live paths |
| [90-second presentation](AUTUMN_RECRUITING_DEMO_SCRIPT.md) | A concrete upload, answer and evidence story |
| [Architecture](ARCHITECTURE.md) | Control flow, data flow and runtime topology |
| [Design decisions](architecture_decisions.md) | Why these boundaries and technologies |
| [Interview evidence](AUTUMN_RECRUITING_EVIDENCE.md) | How to explain the validated capabilities |
| [Resume examples](RESUME_DRAFT.md) | Evidence-backed project descriptions |

## Install, run and validate

| Document | Purpose |
|---|---|
| [Reproducibility](REPRODUCIBILITY.md) | Environment setup and pinned joint evaluation |
| [Validation commands](VALIDATION_COMMANDS.md) | Independent, joint and optional infrastructure checks |
| [Configuration](CONFIGURATION.md) | Runtime and provider settings |
| [Operational limitations](PRODUCTION_LIMITATIONS.md) | Supported boundaries and unverified scenarios |
| [Tenant isolation](MULTI_TENANCY_BOUNDARY.md) | Authorization scope and remaining gaps |
| [Backup and restore](PRODUCTION_BACKUP_RESTORE.md) | Recovery rehearsal and rollback |
| [Observability](OBSERVABILITY.md) | Metrics, privacy and troubleshooting |
| [Encoding](ENCODING.md) | Windows / UTF-8 behavior |

## Subsystems

| Document | Purpose |
|---|---|
| [Hybrid RAG](RAG_MILVUS.md) | Indexing and retrieval |
| [Index leases](RAG_INDEX_LEASE.md) | Worker ownership and recovery |
| [Milvus cutover](MILVUS3_CUTOVER.md) · [BM25 cutover](BM25_CUTOVER.md) | Collection migration and rollback |
| [Qwen3 reranking](QWEN3_RERANK.md) | Controlled rollout and evaluation |
| [Ticker resolution](TICKER_RESOLVE.md) | Company / ticker mapping |
| [Clarification](HITL_CLARIFICATION.md) | Pause and resume contract |
| [Input guardrails](INPUT_GUARDRAIL.md) | Query and upload boundaries |
| [Structured answers](STRUCTURED_ANSWER.md) | Citation protocol and verification |
| [Queue and workers](QUEUE_WORKER_INTEGRATION.md) | Multi-process fault-injection evidence |
| [Provider resilience](PROVIDER_RESILIENCE.md) | Retry, deadlines and failures |
| [MCP](MCP.md) · [MCP layer](../mcp_layer/README.md) | Optional tool interface |

## Evaluation and historical evidence

[FinAgentBench](https://github.com/majiali423/finagentbench-demo) is the product's
separate FinRun evaluator. Current product CI, frozen contract tests and sealed
dataset experiments answer different questions.

| Document | Purpose |
|---|---|
| [Evidence index](EVIDENCE_INDEX.md) | Historical scores, source identities and reports |
| [Product evaluation](evaluation_strategy.md) · [FinAgentBench architecture](https://github.com/majiali423/finagentbench-demo/blob/master/docs/architecture.md) | 24-task candidate gold, B1/B2 lexical diagnostic, `lumenfin_eval_contract.v1` layers; not formal accuracy |
| [FinanceBench](FINANCEBENCH_EVAL.md) | Historical retrieval experiments |
| [LEDGER development boundary](FINANCEBENCH_NEXT_PHASE.md) | Sealed development chain and restrictions |
| [LEDGER index](LEDGER_PUBLIC_HOLDOUT_INDEX.md) · [E2E](LEDGER_PUBLIC_HOLDOUT_E2E.md) | Versioned index and end-to-end evidence |
| [Frozen RC report](PORTFOLIO_RELEASE_REPORT.md) | Evidence for the release tag still used by compatibility CI |

Historical hashes, scores and regression cases are retained. Current source may
have evolved beyond the source version used by a sealed experiment.
Completed phase logs and superseded packaging narratives are available in
[Git history](https://github.com/majiali423/lumenfin-agent/tree/60e4ed6a06d7afb6fce907413d2359cbf89eae44/docs).

[MIT license](../LICENSE) · [Third-party notices](../THIRD_PARTY_NOTICES.md)
