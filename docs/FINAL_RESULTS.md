# Final Results Summary

> Packaging narrative (Before → After). **Authoritative RC numbers** live in
> [`PORTFOLIO_RELEASE_REPORT.md`](PORTFOLIO_RELEASE_REPORT.md) and
> [`../reports/current/PHASE6_FULL_VALIDATION_REPORT.md`](../reports/current/PHASE6_FULL_VALIDATION_REPORT.md).
> Rows below that cite mean **92.97** / live 8/8 are historical packaging
> snapshots under earlier evaluator pins — do not treat them as the current
> `v0.1.0-rc.3` scoreboard.

Before → After packaging summary for LumenFin + FinAgentBench.

---

## Capability table

| Capability | Before | After |
|------------|--------|-------|
| Entity leakage | Peer pollution from filings / expansion | Issuer-only scope; `entity_leakage` gated |
| Financial grounding | PDF runs with 0 checkable AST facts | Issuer SEC/Yahoo fill → verified computable facts |
| Citation / claims | Weak / decorative cites; unbound prose | Claim-bound verified assertions + ledger |
| Fail closed | Partial / could mint numerics on sparse paths | Validated OpenAI + sparse upload-only |
| Benchmark | Manual reading / internal golden only | FinAgentBench FinRun gate (replay + mutations) |
| Multi-company | Fragile entity parity | AAPL–MSFT + NVDA–AMD RC PASS |
| Long document | Unproven under stress | MSFT long 10-K completes with claim coverage |
| Evaluation independence | Coupled to agent internals | Neutral FinRun + adapters |

---

## RC Validation snapshot (latest packaging gate)

| Gate | Result |
|------|--------|
| Live RC pack | **8/8 PASS** |
| Offline (LumenFin unit + FAB unit/regression/correctness) | **PASS** |
| Readiness dimensions | **7/7 PASS** |
| Mean FAB score (completed cases, informational) | **92.97** |
| evidence_coverage (completed FAB) | **100** |
| numeric_correctness (completed FAB) | **100** (floor ≥80 required) |
| entity_leakage (issuer cases) | **100** |

Cases: Apple live, NVIDIA 10-K PDF, Tesla live, Microsoft long 10-K, Apple–Microsoft compare, NVIDIA–AMD compare, OpenAI fail-closed, sparse Oracle upload-only.

---

## Illustrative deltas (phase evidence)

| Phase | Example signal |
|-------|----------------|
| Financial Grounding | NVIDIA PDF: checkable fundamentals recovered (0 → 3 class of gap) |
| Claim Binding | NVIDIA: page markers and verified-in-report coverage strengthened |
| Production Hardening | Long / multi / fail-closed pack **5/5** then RC **8/8** |

Exact run artifacts live under `outputs/` and `finagentbench-demo/outputs/lumenfin_rc_validation/`.

---

## Current limitations (do not hide)

- External API dependency (LLM, embeddings, market/SEC)
- Live-source vs document page-citation difference (`#pN` often 0 on live-only)
- Model availability / rename / quota operational risk
- Limited real-company coverage relative to production universe
- Milvus Lite single-writer constraints
- Not investment advice; demo/research tooling posture

---

## Non-claims

This project does **not** claim perfect factuality, unlimited issuer coverage, or that FinAgentBench scores alone prove investment quality. It claims a **measurable reliability posture** under an external gate.
