# LumenFin RC Production Readiness Assessment

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and regression evidence

Generated: 2026-07-25T12:23:40.064647+00:00

## Executive verdict

**READY for Release Candidate**

- Live RC pack: **8/8**
- Offline gates: **PASS**
- Mean FAB score (completed, informational): **92.97**
- Readiness dimensions: **7/7**

## Dimension checklist

| Dimension | Status | Evidence |
|-----------|:------:|----------|
| Deterministic tests | PASS | LumenFin unit + FinAgentBench unit/regression |
| Issuer numeric grounding | PASS | Checkable formula+inputs on issuer diligence |
| Claim → evidence binding | PASS | Verified claims appear in report with citations |
| Multi-company routing | PASS | AAPL–MSFT and NVDA–AMD entity parity without peer fan-out |
| Long-document stability | PASS | MSFT long 10-K completes with claim coverage |
| Fail-closed honesty | PASS | OpenAI + sparse upload refuse invented fundamentals |
| FinAgentBench floors | PASS | evidence_coverage=100 and numeric_correctness≥80 on completed FAB cases |

## What this RC proves

1. Real-company coverage across Apple, NVIDIA, Tesla, Microsoft, AMD (compare), plus negative controls.
2. Issuer SEC financial grounding + claim→evidence binding remain intact under long PDF and multi-company load.
3. Fail-closed paths do not invent AST-checkable fundamentals or verified numeric claims.
4. FinAgentBench `ci` floors hold without evaluator changes.

## Explicit non-goals (this RC)

- No new claim/citation rules
- No FinAgentBench threshold relaxation
- No retrieval-quality feature expansion

## Residual risks (accept or track — not P0 invent-numbers)

- DeepSeek / DashScope model renames and quota remain operational dependencies.
- Live-only issuers still have 0 `#pN` (fundamentals citations by design).
- Growth claims remain rejected without multi-period fundamentals (honest).
- Milvus Lite single-process lock / AllocTimestamp noise under concurrent local use.

## Go / No-Go

**GO** for Release Candidate: reliability gates green on expanded real-company pack; prior grounding / claim-binding / hardening evidence synthesized; architecture index published.

## Related artifacts

- Final reliability: `LumenFin_RC_Final_Reliability_Report.md`
- Architecture index: `docs/ARCHITECTURE_INDEX.md`
- RC validation JSON: `../finagentbench-demo/outputs/lumenfin_rc_validation/validation.json`
