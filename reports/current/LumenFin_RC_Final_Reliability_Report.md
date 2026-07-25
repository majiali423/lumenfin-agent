# LumenFin RC Final Reliability Report

Generated: 2026-07-25T12:23:40.064647+00:00

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
| Apple live | issuer_live | `completed` | Y | `['Apple']` | 13 | 1.0 | 0 | 3 | 92.97 |
| NVIDIA 10-K PDF | issuer_pdf | `completed` | Y | `['NVIDIA']` | 13 | 1.0 | 36 | 3 | 92.97 |
| Tesla live | issuer_live | `completed` | Y | `['Tesla']` | 13 | 1.0 | 0 | 3 | 92.97 |
| Microsoft long 10-K | long_document | `completed` | Y | `['Microsoft']` | 8 | 1.0 | 13 | 2 | 92.97 |
| Compare Apple vs Microsoft | multi_company | `completed` | Y | `['Apple', 'Microsoft']` | 23 | 1.0 | 0 | 5 | 92.97 |
| Compare NVIDIA vs AMD | multi_company | `completed` | Y | `['NVIDIA', 'AMD']` | 26 | 1.0 | 0 | 6 | 92.97 |
| OpenAI fail-closed | failure_recovery | `incomplete_data` | Y | `['OpenAI']` | 1 | 0.0 | 0 | 0 | None |
| Sparse upload-only fail-closed | failure_recovery | `incomplete_data` | Y | `['Oracle']` | 1 | 0.0 | 2 | 0 | None |

## 3. Claim coverage & failure recovery

### Completed diligence

| Case | Bind rate | Entity claim coverage | Page-anchored | Verified in report |
|------|----------:|----------------------:|--------------:|-------------------:|
| Apple live | 0.9286 | 1.0 | 0 | 13/13 |
| NVIDIA 10-K PDF | 0.9286 | 1.0 | 6 | 13/13 |
| Tesla live | 0.9286 | 1.0 | 0 | 13/13 |
| Microsoft long 10-K | 0.6667 | 1.0 | 1 | 8/8 |
| Compare Apple vs Microsoft | 0.8846 | 1.0 | 0 | 23/23 |
| Compare NVIDIA vs AMD | 0.9286 | 1.0 | 0 | 26/26 |

### Fail-closed

| Case | Status | Checkable | Invented numeric? |
|------|--------|----------:|:-----------------:|
| OpenAI fail-closed | `incomplete_data` | 0 | N |
| Sparse upload-only fail-closed | `incomplete_data` | 0 | N |

## 4. FinAgentBench reliability (completed cases)

| Case | Score | evidence_coverage | evidence_consistency | numeric_correctness | entity_leakage |
|------|------:|------------------:|---------------------:|--------------------:|---------------:|
| Apple live | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| NVIDIA 10-K PDF | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| Tesla live | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| Microsoft long 10-K | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| Compare Apple vs Microsoft | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| Compare NVIDIA vs AMD | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |

## 5. Prior phase evidence (synthesized)

| Phase | Artifact | Present |
|-------|----------|:-------:|
| baseline | `C:\a_project\Projects\lumenfin-agent\LumenFin_Final_Reliability_Baseline.md` | Y |
| grounding | `C:\a_project\Projects\lumenfin-agent\LumenFin_Financial_Grounding_Validation.md` | Y |
| claim_binding | `C:\a_project\Projects\lumenfin-agent\LumenFin_Claim_Evidence_Binding_Report.md` | Y |
| hardening | `C:\a_project\Projects\lumenfin-agent\LumenFin_Production_Hardening_Report.md` | Y |
| e2e | `C:\a_project\Projects\lumenfin-agent\LumenFin_E2E_Audit_Report.md` | Y |
| regression | `C:\a_project\Projects\lumenfin-agent\LumenFin_Regression_Comparison.md` | Y |

Key prior results carried into RC:
- Financial Grounding: NVDA checkable 0→3, numeric 100, issuer-only retained
- Claim Binding: NVDA `#pN` 6→36; verified claims rendered inline (13/13)
- Production Hardening: 5/5 (long MSFT, AAPL–MSFT, long AAPL, OpenAI, sparse)

## 6. Gate detail

### Apple live

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=13 report_cov=1.0
- **PASS** `long_or_metric_stability` — checkable=3 markers=0
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0
- **PASS** `fab_no_entity_leak` — 100.0

### NVIDIA 10-K PDF

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=13 report_cov=1.0
- **PASS** `long_or_metric_stability` — checkable=3 markers=36
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0
- **PASS** `fab_no_entity_leak` — 100.0

### Tesla live

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=13 report_cov=1.0
- **PASS** `long_or_metric_stability` — checkable=3 markers=0
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0
- **PASS** `fab_no_entity_leak` — 100.0

### Microsoft long 10-K

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=8 report_cov=1.0
- **PASS** `long_or_metric_stability` — checkable=2 markers=13
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0
- **PASS** `fab_no_entity_leak` — 100.0

### Compare Apple vs Microsoft

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=23 report_cov=1.0
- **PASS** `per_entity_numeric_claims` — {'Apple': {'verified': 13, 'numeric': 8, 'risk': 4, 'investment': 1}, 'Microsoft': {'verified': 10, 'numeric': 6, 'risk': 4, 'investment': 0}}
- **PASS** `entity_set` — entities=['Apple', 'Microsoft']
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0

### Compare NVIDIA vs AMD

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=26 report_cov=1.0
- **PASS** `per_entity_numeric_claims` — {'NVIDIA': {'verified': 13, 'numeric': 8, 'risk': 4, 'investment': 1}, 'AMD': {'verified': 13, 'numeric': 8, 'risk': 4, 'investment': 1}}
- **PASS** `entity_set` — entities=['NVIDIA', 'AMD']
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0

### OpenAI fail-closed

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=incomplete_data expect=incomplete_data
- **PASS** `fail_closed_no_checkable` — checkable=0
- **PASS** `no_invented_numeric_claims` — {'OpenAI': {'verified': 1, 'numeric': 0, 'risk': 1, 'investment': 0}}
- **PASS** `recovery_completed_path` — status=incomplete_data

### Sparse upload-only fail-closed

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=incomplete_data expect=incomplete_data
- **PASS** `fail_closed_no_checkable` — checkable=0
- **PASS** `no_invented_numeric_claims` — {'Oracle': {'verified': 1, 'numeric': 0, 'risk': 1, 'investment': 0}}
- **PASS** `recovery_completed_path` — status=incomplete_data

## 7. Verdict

**RC reliability gate: PASS.**

## Artifacts

- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_rc_validation\validation.json`
- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_rc_validation\offline_gates.json`
- FinRuns: `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_rc_validation\finrun`
