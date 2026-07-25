# LumenFin RC Final Reliability Report

Generated: 2026-07-25T16:24:44+00:00 (live pack)
Frozen for release evidence: 2026-07-26

Release Candidate validation of **current** LumenFin + FinAgentBench.
No new claim/citation rules. No evaluator threshold changes.

Canonical path: `LumenFin → export_finrun_state() → FinAgentBench (ci)`.

## Candidate code state (live pack)

| Item | Value |
|------|-------|
| LumenFin HEAD at live run | `2e28d74` (`docs(release): record clean-clone validation evidence`) |
| FinAgentBench HEAD at live run | `6700846` (`docs(release): record clean-clone validation evidence`) |
| Working-tree delta at live run | process `DEEPSEEK_API_KEY` cleared so AppConfig loaded project `.env` (no Agent/case/fixture edits) |
| Live runner | `finagentbench-demo/scripts/run_rc_validation.py` |
| Exit | `LIVE_RC_EXIT=0` |

Post-live code HEADs (no Agent/case/fixture/threshold edits): LumenFin `0f895f8` (env fail-fast), FinAgentBench `f58e479` (live-RC fallback abort). See `reports/Clean_Clone_Validation_Report.md`.

## Provider honesty

| Item | Observed |
|------|----------|
| Provider | `deepseek` (all 8 cases) |
| Model | `deepseek-v4-flash` (all 8 cases) |
| HTTP 401 (DeepSeek chat) | `0` |
| `local-fallback` count | `0` |
| DeepSeek chat HTTP 200 | ~40 |

## 1. Offline gates

| Gate | OK | Return code |
|------|:--:|------------:|
| `lumenfin_unit` | Y | 0 |
| `finagentbench_unit` | Y | 0 |
| `finagentbench_lumenfin_regression` | Y | 0 |
| `finagentbench_correctness` | Y | 0 |

## 2. Expanded real-company RC pack

| Cases | Judgment |
|------:|----------|
| 8 | **8/8** |
| completed | **6** |
| expected fail-closed | **2** |

| Case | Scenario | Status | OK | Entities | Verified claims | Report cov | Checkable | FAB score |
|------|----------|--------|:--:|----------|----------------:|-----------:|----------:|----------:|
| Apple live | issuer_live | `completed` | Y | Apple | 13 | 1.0 | 3 | 92.97 |
| NVIDIA 10-K PDF | issuer_pdf | `completed` | Y | NVIDIA | 10 | 1.0 | 2 | 92.97 |
| Tesla live | issuer_live | `completed` | Y | Tesla | 13 | 1.0 | 3 | 92.97 |
| Microsoft long 10-K | long_document | `completed` | Y | Microsoft | 10 | 1.0 | 2 | 92.97 |
| Compare Apple vs Microsoft | multi_company | `completed` | Y | Apple, Microsoft | 23 | 1.0 | 5 | 92.97 |
| Compare NVIDIA vs AMD | multi_company | `completed` | Y | NVIDIA, AMD | 26 | 1.0 | 6 | 92.97 |
| OpenAI fail-closed | failure_recovery | `incomplete_data` | Y | OpenAI | 1 | 0.0 | 0 | — |
| Sparse upload-only fail-closed | failure_recovery | `incomplete_data` | Y | Oracle | 1 | 0.0 | 0 | — |

### Floors (completed cases)

| Metric | Floor |
|--------|------:|
| entity_leakage | 100 |
| numeric_correctness | 100 |
| evidence_coverage | 100 |
| Completed-case FAB mean | **92.97** |

### Fail-closed honesty

OpenAI and sparse cases: `incomplete_data`, judgment ok, checkable=0, no invented numeric claims.

## 3. Verdict

**RC reliability gate: PASS.**
**LIVE_RC_EXIT=0.**

## Artifacts (local, not committed)

- `finagentbench-demo/outputs/lumenfin_rc_validation/validation.json`
- `finagentbench-demo/outputs/lumenfin_rc_validation/offline_gates.json`
- case state / finrun JSON under the same outputs tree

## Prior phase evidence

| Phase | Artifact | Present |
|-------|----------|:-------:|
| grounding | `reports/history/LumenFin_Financial_Grounding_Validation.md` | Y |
| claim_binding | `reports/history/LumenFin_Claim_Evidence_Binding_Report.md` | Y |
| hardening | `reports/history/LumenFin_Production_Hardening_Report.md` | Y |
| e2e | `reports/history/LumenFin_E2E_Audit_Report.md` | Y |
| regression | `reports/history/LumenFin_Regression_Comparison.md` | Y |
| rc_current | `reports/current/LumenFin_RC_Final_Reliability_Report.md` | Y |

This report intentionally omits API keys, Authorization headers, `.env` contents, and user-home absolute paths.
