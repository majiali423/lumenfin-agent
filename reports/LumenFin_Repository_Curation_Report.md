# LumenFin Repository Curation Report

Date: 2026-07-25
Judgment: **NOT READY — MANUAL REVIEW REQUIRED**

## 1. FinAgentBench migration note

FAB `RM` archive migration was completed in the sibling repo. LumenFin RC entrypoints remain in FinAgentBench (`run_rc_validation.py`, `rc_runtime.py`, `validate_cross_repo.py`).

## 2. Secret risk

| Secret Risk | Repository | Git Tracked/Untracked | Required Action |
|-------------|------------|-----------------------|-----------------|
| `DEEPSEEK_API_KEY` likely real in local `.env` | LumenFin | **untracked / gitignored**; never in git history | **Rotate out-of-band** (not printed here) |
| `DASHSCOPE_API_KEY` likely real in local `.env` | LumenFin | untracked / gitignored | **Rotate out-of-band** |
| `ALPHAVANTAGE_API_KEY` likely real in local `.env` | LumenFin | untracked / gitignored | **Rotate out-of-band** |
| `.env.example` | LumenFin | tracked placeholders / URLs only | keep |
| Keys in git history | LumenFin | none for `.env` | no history scrub required for `.env` path |

Local `.env` was not deleted. Provider rotation was not invoked automatically.

If these values were ever pasted into chat/logs: treat as **SECURITY BLOCKER — ROTATE CREDENTIAL**.

## 3. Deleted files

No production deletions. Temporary helper scripts used during curation were removed after use.

## 4. Archived files (staged)

- `tools/archived_audits/*` historical runners
- `reports/history/*` superseded engineering reports
- `reports/current/*` formal RC evidence copies

## 5. Ignored

Confirmed via `.gitignore`: `.env`, `outputs/`, `data/**`, Milvus, `.venv`, `fixtures/e2e_real/`, `.local-fixtures/`, root generated `LumenFin_*.md`, local interview companions.

## 6. SEC fixture handling

| Item | Action |
|------|--------|
| `tests/fixtures/sec/minimal/*.html` | staged (minimized) |
| `tests/fixtures/sec/sources/*.txt` | staged extracts |
| `tests/fixtures/sec/derived/*.pdf` | built + staged; labeled non-official / HTML-derived |
| `tests/fixtures/sec/manifest.json` | SHA-256 filled for committed paths |
| `fixtures/e2e_real/**` | ignored / not staged |
| `fixtures/stress/MANIFEST.json` | absolute path removed (`fixtures/stress`) |
| stress regenerated PDFs | left untracked |

## 7. README / docs fixes

- Python pin clarified as **3.12** (CI)
- lockfile install path retained
- `docs/VALIDATION_COMMANDS.md` added
- PyMuPDF public-Docker blocker documented
- Interview notes moved to optional `docs/portfolio/` (not staged)
- Broken pitch/command-guide links removed from published index

## 8. Supported commands

See `docs/VALIDATION_COMMANDS.md` (LumenFin tests + sibling FAB gates).

## 9. Explicit staging groups

See `reports/LumenFin_Commit_Plan.md`.

Currently staged: docs/report/archive/SEC fixture curation subset.
**Not yet staged:** majority of `src/lumenfin/**`, new tests, MCP, CI/Docker, lockfile — still dirty in working tree for owner-reviewed grouped commits.

## 10. Staged audit (curation subset)

- `git diff --cached --check`: PASS after trailing-whitespace normalization
- No high-confidence secrets in staged diff
- Derived PDFs are small (<150KB)
- No `git add .`

## 11. Offline validation

| Gate | Result |
|------|--------|
| `python scripts/run_tests.py` | 267 PASS, 1 skipped |
| concurrency / HITL | included in suite |
| FAB unittest | 77 PASS |
| mutation | 4/4 |
| cross-repo gate | PASS |
| RC import / dry-run | PASS |

## 12. Clean-clone

Not completed (production source still largely unstaged; no temp commit/worktree clone).

## 13. Live RC

Not re-run. Prerequisites incomplete (clean-clone, full explicit stage of runtime commits, owner auth).

## 14. Uncertain

- Stage vs ignore `docs/portfolio/INTERVIEW_NOTES.md`
- Stage vs leave untracked `reports/LumenFin_{Cleanup_Plan,Repository_Inventory}.md`
- Whether synthetic stress PDFs should be committed or only generated
- Public LICENSE still intentionally absent
- FAB pin tag `v0.1.0-rc.1` still missing remotely
- Docker daemon previously unavailable

## 15. Judgment

**NOT READY — MANUAL REVIEW REQUIRED**

Offline quality gates are green, but clean-commit readiness still needs:

1. Credential rotation confirmation
2. Explicit grouped staging of remaining `src/` + `tests/` + CI
3. Clean-clone proof
4. Owner-authorized commits (FAB first)
5. Live RC only after the above
