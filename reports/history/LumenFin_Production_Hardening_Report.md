# LumenFin Production Hardening Report

Status: Historical
Superseded by: `../current/LumenFin_Post_RC_Hardening_Report.md`
Purpose: Engineering evolution and regression evidence

Generated: 2026-07-24T17:59:30.157119+00:00

Scope: **claim coverage**, **failure recovery**, and **reliability** under long-document, multi-company, and multi-metric stress — **without adding new claim rules**.

Path: live LumenFin → `export_finrun_state()` → FinAgentBench (`ci`). Evaluators unchanged.

## 1. Summary

| Cases | Passed |
|------:|-------:|
| 5 | **5/5** |

| Case | Scenario | Status | OK | Verified claims | Report coverage | #pN | Checkable | FAB score | Elapsed ms |
|------|----------|--------|:--:|----------------:|----------------:|----:|----------:|----------:|-----------:|
| Long document — Microsoft 10-K | long_document | `completed` | Y | 10 | 1.0 | 13 | 2 | 92.97 | 95304.1 |
| Multi-company — Apple vs Microsoft | multi_company | `completed` | Y | 23 | 1.0 | 0 | 5 | 92.97 | 37245.4 |
| Multi-metric long Apple 10-K | multi_metric | `completed` | Y | 10 | 1.0 | 13 | 3 | 92.97 | 87107.9 |
| Failure recovery — OpenAI fail-closed | failure_recovery | `incomplete_data` | Y | 1 | 0.0 | 0 | 0 | None | 11618.8 |
| Failure recovery — sparse upload-only | failure_recovery | `incomplete_data` | Y | 1 | 0.0 | 2 | 0 | None | 26903.0 |

## 2. Claim Coverage

| Case | Bind rate | Entities w/ numeric claims | Entity claim coverage | Page-anchored verified | Verified citations in report |
|------|----------:|---------------------------:|----------------------:|-----------------------:|-----------------------------:|
| Long document — Microsoft 10-K | 0.8333 | 1 | 1.0 | 1 | 10/10 |
| Multi-company — Apple vs Microsoft | 0.8846 | 2 | 1.0 | 0 | 23/23 |
| Multi-metric long Apple 10-K | 0.7143 | 1 | 1.0 | 1 | 10/10 |

### Multi-company per-entity claims

```json
{
  "Apple": {
    "verified": 13,
    "numeric": 8,
    "risk": 4,
    "investment": 1
  },
  "Microsoft": {
    "verified": 10,
    "numeric": 6,
    "risk": 4,
    "investment": 0
  }
}
```

## 3. Failure Recovery

| Case | Expected | Got | No crash | No invented numeric claims | Checkable=0 |
|------|----------|-----|:--------:|:--------------------------:|:-----------:|
| Failure recovery — OpenAI fail-closed | `incomplete_data` | `incomplete_data` | Y | Y | Y |
| Failure recovery — sparse upload-only | `incomplete_data` | `incomplete_data` | Y | Y | Y |

## 4. Reliability (FinAgentBench — completed cases)

| Case | Score | evidence_coverage | evidence_consistency | numeric_correctness | entity_leakage |
|------|------:|------------------:|---------------------:|--------------------:|---------------:|
| Long document — Microsoft 10-K | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| Multi-company — Apple vs Microsoft | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |
| Multi-metric long Apple 10-K | 92.97 | 100.0 | 100.0 | 100.0 | 100.0 |

## 5. Gate Details

### Long document — Microsoft 10-K

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=10 report_cov=1.0
- **PASS** `long_or_metric_stability` — checkable=2 markers=13
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0
- **PASS** `fab_no_entity_leak` — 100.0

### Multi-company — Apple vs Microsoft

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=23 report_cov=1.0
- **PASS** `per_entity_numeric_claims` — {'Apple': {'verified': 13, 'numeric': 8, 'risk': 4, 'investment': 1}, 'Microsoft': {'verified': 10, 'numeric': 6, 'risk': 4, 'investment': 0}}
- **PASS** `entity_set` — entities=['Apple', 'Microsoft']
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0

### Multi-metric long Apple 10-K

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=completed expect=completed
- **PASS** `claim_coverage_min` — verified=10 report_cov=1.0
- **PASS** `long_or_metric_stability` — checkable=3 markers=13
- **PASS** `fab_evidence_coverage` — 100.0
- **PASS** `fab_numeric` — 100.0
- **PASS** `fab_no_entity_leak` — 100.0

### Failure recovery — OpenAI fail-closed

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=incomplete_data expect=incomplete_data
- **PASS** `fail_closed_no_checkable` — checkable=0
- **PASS** `no_invented_numeric_claims` — {'OpenAI': {'verified': 1, 'numeric': 0, 'risk': 1, 'investment': 0}}
- **PASS** `recovery_completed_path` — status=incomplete_data

### Failure recovery — sparse upload-only

- **PASS** `no_crash` — ok
- **PASS** `expected_workflow` — got=incomplete_data expect=incomplete_data
- **PASS** `fail_closed_no_checkable` — checkable=0
- **PASS** `no_invented_numeric_claims` — {'Oracle': {'verified': 1, 'numeric': 0, 'risk': 1, 'investment': 0}}
- **PASS** `recovery_completed_path` — status=incomplete_data

## 6. Verdict

**PASS — production hardening suite green.**

Hardening focus remains: long-doc claim coverage, multi-entity claim parity, and fail-closed recovery under live APIs.

## Artifacts

- `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_production_hardening\validation.json`
- FinRuns: `C:\a_project\Projects\finagentbench-demo\outputs\lumenfin_production_hardening\finrun`
