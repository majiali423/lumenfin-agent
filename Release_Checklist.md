# Release Checklist

Target: LumenFin + FinAgentBench `0.1.0rc1`

| Item | Status | Evidence |
|------|:------:|----------|
| Secrets removed | PASS | No committed key/token pattern; `.env` ignored; blank examples |
| Production Compose fail-closed | PASS | production/live forced; secrets required; data-service ports private |
| Local DB/output/log/cache excluded | PASS | `.gitignore`; `git ls-files` found none |
| Dependencies locked | PASS | `requirements-lock.txt`; CI/Docker consume it |
| FinRun contract versioned | PASS | schema `1.0`; unknown versions rejected |
| FinAgentBench tagged | **BLOCKED** | No existing tags; dirty worktree must be committed first |
| LumenFin pinned to FAB ref | PREPARED | CI defaults to `v0.1.0-rc.1`; tag must exist first |
| LumenFin offline tests | PASS | 265 tests, 1 skipped |
| FinAgentBench offline tests | PASS | 75 tests |
| Mutation 4/4 | PASS | release mutation report, detection rate 1.0 |
| Correctness validation | PASS | all expected mutations detected |
| Cross-repo gate | PASS | FinRun `1.0`, profile `ci`, both commits recorded |
| Offline demo | PASS | baseline + 4 mutations, no keys |
| Live RC 8/8 | PASS | completed score mean 92.97; negative controls fail closed |
| Entity/numeric/evidence floors | PASS | completed cases 100/100/100 floors |
| Linter | PASS | no new diagnostics |
| Docker build | **BLOCKED** | Docker daemon unavailable on validation host |
| Public distribution license | **BLOCKED** | Repository owner must select license/third-party notice policy |
| Documentation complete | PASS | configuration/reproducibility/limits/demo + FAB metric/CI docs |
| Known limitations documented | PASS | `docs/PRODUCTION_LIMITATIONS.md` |
| Clean release worktrees | **BLOCKED** | both repositories contain uncommitted release changes |

## Required closing actions

1. Review and commit FinAgentBench release content.
2. Run its GitHub Actions and create `v0.1.0-rc.1`.
3. Start Docker and run `docker build -t lumenfin:0.1.0rc1 .`.
4. Select a license before any public release.
5. Review and commit LumenFin after the FAB tag is resolvable.
6. Regenerate cross-repository evidence from clean trees.
7. Run LumenFin CI/live RC and create its `v0.1.0-rc.1` tag.

## Current decision

**NO-GO — Blockers Remain**

The code/test evidence supports an RC, but an actual release must not be cut
from dirty, untagged trees or without the claimed Docker build verification.
