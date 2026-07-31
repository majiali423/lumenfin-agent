# LumenFin RC Final Reliability Report

Generated: 2026-07-31T10:55:28.304137+00:00

Release Candidate validation of **current** LumenFin + FinAgentBench.
No new claim/citation rules. No evaluator threshold changes.

Canonical path: `LumenFin → export_finrun_state() → FinAgentBench (ci)`.

## 1. Offline gates

| Gate | OK | Return code |
|------|:--:|------------:|
| `lumenfin_unit` | Y | 0 |
| `finagentbench_unit` | Y | 0 |
| `finagentbench_lumenfin_regression` | Y | 0 |
| `finagentbench_correctness` | Y | 0 |

## 2. Expanded real-company RC pack

| Cases | Passed |
|------:|-------:|
| 8 | **8/8** |

| Case | Scenario | Status | OK | Entities | Verified claims | Report cov | #pN | Checkable | FAB score |
|------|----------|--------|:--:|----------|----------------:|-----------:|----:|----------:|----------:|
| Apple live | issuer_live | `completed` | Y | `['Apple']` | 13 | 1.0 | 0 | 3 | 100.0 |
| NVIDIA 10-K PDF | issuer_pdf | `completed` | Y | `['NVIDIA']` | 10 | 1.0 | 29 | 2 | 100.0 |
| Tesla live | issuer_live | `completed` | Y | `['Tesla']` | 13 | 1.0 | 0 | 3 | 100.0 |
| Microsoft long 10-K | long_document | `completed` | Y | `['Microsoft']` | 8 | 1.0 | 6 | 2 | 100.0 |
| Compare Apple vs Microsoft | multi_company | `completed` | Y | `['Apple', 'Microsoft']` | 23 | 1.0 | 0 | 5 | 100.0 |
| Compare NVIDIA vs AMD | multi_company | `completed` | Y | `['NVIDIA', 'AMD']` | 24 | 1.0 | 0 | 6 | 100.0 |
| OpenAI fail-closed | failure_recovery | `incomplete_data` | Y | `['OpenAI']` | 1 | 0.0 | 0 | 0 | None |
| Sparse upload-only fail-closed | failure_recovery | `incomplete_data` | Y | `['Oracle']` | 1 | 0.0 | 2 | 0 | None |

## 3. Verdict

**RC reliability gate: PASS.**

## Artifacts

- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_rc_validation\validation.json`
- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_rc_validation\offline_gates.json`

## Prior phase evidence

| Phase | Artifact | Present |
|-------|----------|:-------:|
| baseline | `C:\a_project\Projects\lumenfin-agent\LumenFin_Final_Reliability_Baseline.md` | Y |
| grounding | `C:\a_project\Projects\lumenfin-agent\reports\history\LumenFin_Financial_Grounding_Validation.md` | Y |
| claim_binding | `C:\a_project\Projects\lumenfin-agent\reports\history\LumenFin_Claim_Evidence_Binding_Report.md` | Y |
| hardening | `C:\a_project\Projects\lumenfin-agent\reports\history\LumenFin_Production_Hardening_Report.md` | Y |
| e2e | `C:\a_project\Projects\lumenfin-agent\reports\history\LumenFin_E2E_Audit_Report.md` | Y |
| regression | `C:\a_project\Projects\lumenfin-agent\reports\history\LumenFin_Regression_Comparison.md` | Y |
| rc_current | `C:\a_project\Projects\lumenfin-agent\reports\current\LumenFin_RC_Final_Reliability_Report.md` | Y |
