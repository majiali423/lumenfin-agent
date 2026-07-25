# LumenFin RC Production Readiness Assessment

Generated: 2026-07-25T16:24:44+00:00
Frozen for release evidence: 2026-07-26

## Executive verdict

**READY for Release Candidate** (live reliability)

| Gate | Result |
|------|--------|
| Live RC pack | **8/8** judgment |
| Completed / fail-closed | **6** / **2** |
| Provider / model | `deepseek` / `deepseek-v4-flash` |
| HTTP 401 | **0** |
| local-fallback | **0** |
| Offline gates | **PASS** |
| Mean FAB (completed) | **92.97** |
| Entity / numeric / evidence floors | **100 / 100 / 100** |
| `LIVE_RC_EXIT` | **0** |

## Evidence pointers

- Reliability detail: `reports/current/LumenFin_RC_Final_Reliability_Report.md`
- Local validation JSON (untracked outputs): `finagentbench-demo/outputs/lumenfin_rc_validation/validation.json`
- Clean-clone / final HEAD: `reports/Clean_Clone_Validation_Report.md`
- Release blockers outside live reliability (tag, Docker daemon, license): `Release_Checklist.md`

## Scope note

Live Agent behavior, cases, fixtures, metrics, and thresholds were not changed to obtain these scores.
Post-live hardening commits only address credential-env fail-fast and live-RC fallback abort.
