# Release Checklist

> **Historical `0.1.0rc1` checklist — not the current release gate.**
>
> Current development candidate: LumenFin **`0.1.0rc4`** (not yet tagged)
> Latest published candidate: LumenFin **`0.1.0rc3` / `v0.1.0-rc.3`**
> Authority: [`docs/PORTFOLIO_RELEASE_REPORT.md`](docs/PORTFOLIO_RELEASE_REPORT.md)  
> Full validation: [`docs/PRODUCTION_LIMITATIONS.md`](docs/PRODUCTION_LIMITATIONS.md)
>
> The tables below are retained only as an audit snapshot of the earlier rc1
> closure (license/push were still blocked then). Do **not** treat BLOCKED /
> NOT DONE rows as today's status.

Target (historical): LumenFin + FinAgentBench `0.1.0rc1`

| Item | Status (rc1 era) | Evidence |
|------|:------:|----------|
| Secrets removed | PASS | No committed key/token pattern; `.env` ignored; blank examples |
| Env conflict fail-fast | PASS | `env_bootstrap` + `tests.test_env_bootstrap_conflicts` |
| Production Compose fail-closed | PASS | production/live forced; secrets required; data-service ports private |
| Local DB/output/log/cache excluded | PASS | `.gitignore`; `git ls-files` found none |
| Dependencies locked | PASS | `requirements-lock.txt`; CI/Docker consume it |
| FinRun contract versioned | PASS | schema `1.0`; unknown versions rejected |
| FinAgentBench tagged | **BLOCKED (then)** | Tag/push not authorized in that pass |
| LumenFin pinned to FAB ref | PREPARED (then) | CI defaulted to `v0.1.0-rc.1` |
| LumenFin offline tests | PASS | 271 tests, 1 skipped (rc1 HEAD) |
| FinAgentBench offline tests | PASS | 78 tests (1 skipped) |
| Mutation 4/4 | PASS | clean worktree mutation suite |
| Correctness / offline demo | PASS | baseline + mutations detected |
| Cross-repo gate | PASS | FinRun `1.0`, profile `ci`, dirty=false |
| Live RC 8/8 | PASS | deepseek / deepseek-v4-flash; FAB mean 92.97 under pin `v0.1.0-rc.1` |
| Documentation complete | PASS | RC reports + configuration/reproducibility/limits |
| Known limitations documented | PASS | `docs/PRODUCTION_LIMITATIONS.md` |
| Public distribution license | **BLOCKED (then)** | Owner had not selected license yet |
| Push / tag / GitHub Release | **NOT DONE (then)** | Explicitly out of scope for that pass |

## Current published status (read this)

| Item | Status |
|------|--------|
| Current LumenFin candidate | `0.1.0rc4` on `main`; validation/tag pending |
| License | MIT (`LICENSE`) + `THIRD_PARTY_NOTICES.md` |
| LumenFin tag | `v0.1.0-rc.3` published |
| FinAgentBench package tag | `v0.1.0-rc.4` published |
| LumenFin evaluator pin | FinAgentBench `v0.1.0-rc.3` |
| Public Docker image | **Not** published (AGPL/RSAL boundary) |
