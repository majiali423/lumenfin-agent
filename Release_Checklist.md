# Release Checklist

Target: LumenFin + FinAgentBench `0.1.0rc1`

| Item | Status | Evidence |
|------|:------:|----------|
| Secrets removed | PASS | No committed key/token pattern; `.env` ignored; blank examples |
| Env conflict fail-fast | PASS | `env_bootstrap` + `tests.test_env_bootstrap_conflicts` |
| Production Compose fail-closed | PASS | production/live forced; secrets required; data-service ports private |
| Local DB/output/log/cache excluded | PASS | `.gitignore`; `git ls-files` found none |
| Dependencies locked | PASS | `requirements-lock.txt`; CI/Docker consume it |
| FinRun contract versioned | PASS | schema `1.0`; unknown versions rejected |
| FinAgentBench tagged | **BLOCKED** | Tag/push not authorized in this pass |
| LumenFin pinned to FAB ref | PREPARED | CI defaults to `v0.1.0-rc.1`; tag must exist first |
| LumenFin offline tests | PASS | 271 tests, 1 skipped (final HEAD clean worktree) |
| FinAgentBench offline tests | PASS | 78 tests (1 skipped) |
| Mutation 4/4 | PASS | clean worktree mutation suite |
| Correctness / offline demo | PASS | baseline + mutations detected |
| Cross-repo gate | PASS | FinRun `1.0`, profile `ci`, dirty=false |
| Live RC 8/8 | PASS | deepseek / deepseek-v4-flash; 401=0; fallback=0; FAB mean 92.97 |
| Entity/numeric/evidence floors | PASS | completed cases 100/100/100 |
| Live RC fallback abort | PASS | RC runner forces `allow_local_fallback=False` and aborts on fallback |
| Final HEAD clean-clone | PASS | `reports/Clean_Clone_Validation_Report.md` |
| Tracked fixtures modified | PASS | No fixture changes in post-live commits |
| Docker build | **BLOCKED** | Docker daemon / public publish not in this pass |
| Public distribution license | **BLOCKED** | Owner must select license before public release |
| Documentation complete | PASS | RC reports + configuration/reproducibility/limits |
| Known limitations documented | PASS | `docs/PRODUCTION_LIMITATIONS.md` |
| Clean release worktrees | PASS | post-docs commit (ignored local data only) |
| Push / tag / GitHub Release | **NOT DONE** | Explicitly out of scope |

## Remaining before a public RC cut

1. Owner creates FinAgentBench `v0.1.0-rc.1` after push review.
2. Resolve Docker build evidence on a host with Docker.
3. Select a license before any public distribution.
4. Push is a separate authorized step — not performed here.

## Current decision

**READY FOR PUSH REVIEW**

Live reliability and final-HEAD offline clean-clone gates are green. Remote push, tag, and public Docker/license remain owner-controlled blockers outside this local finalize pass.
