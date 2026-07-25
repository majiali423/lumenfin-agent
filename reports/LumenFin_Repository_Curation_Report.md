# LumenFin Repository Curation Report

Date: 2026-07-26
Judgment: **READY FOR PUSH REVIEW**

## 1. FinAgentBench migration note

FAB archive migration completed. LumenFin RC entrypoints remain in FinAgentBench (`run_rc_validation.py`, `rc_runtime.py`, `validate_cross_repo.py`).

## 2. Secret risk

| Secret Risk | Repository | Git Tracked/Untracked | Required Action |
|-------------|------------|-----------------------|-----------------|
| Local `.env` credentials | LumenFin | **untracked / gitignored** | Keep local; rotate out-of-band if ever exposed |
| `.env.example` | LumenFin | tracked placeholders only | keep |
| Keys in git history | LumenFin | none for `.env` | no scrub required |

Process-env vs `.env` conflicts now fail fast (`src/lumenfin/env_bootstrap.py`); diagnostics print source + length only.

## 3. Curation file decisions (this finalize pass)

| Path | Decision | Reason | Canonical Replacement |
|------|----------|--------|------------------------|
| `reports/LumenFin_Cleanup_Plan.md` | Delete | Intermediate execution plan | `reports/LumenFin_Repository_Curation_Report.md` |
| `reports/LumenFin_Repository_Inventory.md` | Delete | Pre-commit inventory snapshot | This curation report + Commit Plan history |
| `docs/portfolio/` | Exclude (gitignored) | Optional interview notes | `docs/INTERVIEW_NOTES.md` if published later |
| `fixtures/stress/apple_msft_fy2025_table*.pdf`, `tsmc_fy2025_table.pdf` | Exclude (gitignored) | Regenerated local stress PDFs | tracked stress corpus + `build_table_pdf_fixtures.py` |

## 4. Post-live commits

| SHA | Subject |
|-----|---------|
| `0f895f8` | `fix(config): fail fast on conflicting provider credentials` |
| *(docs)* | `docs(release): record successful eight-case live RC` |

## 5. Validation

| Gate | Result |
|------|--------|
| Live RC | 8/8; deepseek; fallback 0; 401 0; FAB mean 92.97 |
| Final HEAD clean worktree tests | 271 OK (1 skip) |
| FAB sibling gates | 78 / mutation 4/4 / cross-repo PASS / RC dry-run PASS |

## 6. Judgment

**READY FOR PUSH REVIEW** — no uncertain tracked candidates remain. Push/tag/release remain unauthorized here.
