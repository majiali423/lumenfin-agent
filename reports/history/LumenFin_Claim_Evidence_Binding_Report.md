# LumenFin Claim → Evidence Binding Report

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and regression evidence

Generated: 2026-07-24T17:45:47.425676+00:00

Path: `claim_binder` → synthesizer (verified claims only) → `export_finrun_state()` → FinAgentBench (`ci`).
Evaluators unchanged. Citations are structural (no prompt-forced citation generation).

## 1. Current Score

| Case | Before score | After score | Before #pN | After #pN | Verified claims | Verified citations in report |
|------|-------------:|------------:|-----------:|----------:|----------------:|-----------------------------:|
| NVIDIA 10-K | 92.97 | **92.97** | 6 | **36** | 13 | 13 |
| Apple live | 92.97 | **92.97** | 0 | **0** | 13 | 13 |

## 2. Citation / Evidence Metrics (After)

| Case | evidence_coverage | evidence_consistency | retrieval_provenance | numeric_correctness | entity_leakage |
|------|------------------:|---------------------:|---------------------:|--------------------:|---------------:|
| NVIDIA | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| Apple | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |

## 3. Before / After Comparison

| Dimension | Before | After | Read |
|-----------|--------|-------|------|
| NVIDIA claim→citation in body | appendix #pN only / unbound metrics | verified claims rendered with [citation] inline | Structural binding |
| NVIDIA bind_rate | ~0 (no claim objects) | 0.9286 | Internal claim filter |
| Apple live citations | 0 `#pN` (no upload) | 0 `#pN`; fundamentals citations on verified claims | Expected for live-only |
| Growth claims | heuristic possible | rejected without multi-period fundamentals | Fail-closed |
| Investment conclusions | template prose | verified composition from numeric+risk only | Fail-closed |

## 4. Claim Binding Detail

- NVIDIA: `{"total_claims": 14, "verified_claims": 13, "rejected_claims": 1, "page_anchored_verified": 6, "by_type": {"numeric": {"total": 8, "verified": 8, "rejected": 0}, "growth": {"total": 1, "verified": 0, "rejected": 1}, "risk_conclusion": {"total": 4, "verified": 4, "rejected": 0}, "investment_conclusion": {"total": 1, "verified": 1, "rejected": 0}}, "bind_rate": 0.9286}`
- Apple: `{"total_claims": 14, "verified_claims": 13, "rejected_claims": 1, "page_anchored_verified": 0, "by_type": {"numeric": {"total": 8, "verified": 8, "rejected": 0}, "growth": {"total": 1, "verified": 0, "rejected": 1}, "risk_conclusion": {"total": 4, "verified": 4, "rejected": 0}, "investment_conclusion": {"total": 1, "verified": 1, "rejected": 0}}, "bind_rate": 0.9286}`

## 5. Gate

- NVIDIA verified claim binding: **PASS**
- Apple verified claim binding: **PASS**

## Artifacts

- `../finagentbench-demo/outputs/lumenfin_claim_binding/validation.json`
