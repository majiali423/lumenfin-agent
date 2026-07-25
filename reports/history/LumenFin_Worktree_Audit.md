# LumenFin Worktree Audit

Status: Historical
Superseded by: `../current/LumenFin_Final_Release_Report.md`
Purpose: Engineering evolution and release-boundary evidence

Date: 2026-07-25
Branch: `main`
HEAD before release commits: `f13ec3d867fa53de0594a6a8c992e9c2ba1e6f6f`
Local tags: none
Remote tags: none

Audit commands: `git status --short`, `git diff --stat`,
`git diff --name-status`, `git ls-files --others --exclude-standard`.

At audit start: 81 modified tracked files and 79 untracked path groups. The
tracked diff contained 6,156 insertions and 1,223 deletions across 82 files.

## Classification

| File/directory | Git status | Category | Recommended action | Reason |
|----------------|------------|----------|--------------------|--------|
| `src/lumenfin/**` | modified + untracked | production source | stage by subsystem | RC Agent runtime, grounding, claims, FinRun, RAG, provider resilience |
| `mcp_layer/**` | modified + untracked | production source | stage separately | Optional MCP boundary and scope enforcement |
| `run_demo.py`, `start_api.py` | modified | production entrypoints | stage | Apply configured RAG profile and package API |
| `tests/**` | modified + untracked | tests | stage with related source | 265+ regression coverage including concurrency/fail-closed/security |
| `.github/workflows/test.yml` | modified | CI/configuration | stage | Locked install and pinned FinAgentBench ref |
| `.env.example` | modified | safe configuration template | stage | Blank secrets and documented production requirements |
| `.gitignore`, `.dockerignore` | modified/tracked | CI/configuration | stage | Excludes secrets, generated state and local reports |
| `Dockerfile`, `docker-compose.yml` | modified | Docker/security configuration | stage | Locked dependencies; production/live fail-fast Compose |
| `pyproject.toml`, `requirements.txt`, `requirements-lock.txt` | modified + untracked | dependency/version configuration | stage | `0.1.0rc1` and reproducible dependency set |
| `README.md`, `CHANGELOG.md` | modified + untracked | documentation/version | stage | RC positioning and history |
| `docs/{FINAL_ARCHITECTURE,CONFIGURATION,REPRODUCIBILITY,PRODUCTION_LIMITATIONS,DEMO_GUIDE,ARCHITECTURE_INDEX}.md` | untracked | release documentation | stage | Required release package |
| Other tracked/untracked `docs/*.md` shown by status | modified + untracked | engineering documentation | stage after link/path review | Architecture, RAG, HITL, evolution and validation rationale |
| `reports/{LumenFin_Final_Release_Report,Joint_Compatibility_Report,LumenFin_Worktree_Audit,LICENSE_RECOMMENDATION}.md` | untracked | formal release reports | stage after final refresh | Canonical audit/release evidence |
| `Release_Checklist.md` | untracked | formal release report | stage after final refresh | Blocking checklist |
| `scripts/run_tests.py`, `scripts/repo_paths.py`, `scripts/run_rag_eval.py`, `scripts/run_rag_index_worker.py` | modified + untracked | release/operations tooling | stage | Reproducible tests and RAG operation |
| `scripts/run_{e2e_acceptance,e2e_production_audit,live_quality_audit,live_showcase,live_table_phrasing_qa,live_zh_table_phrasing_qa}.py` | untracked | validation/demo tooling | stage in tooling commit | RC evidence and interview demos |
| `scripts/{build_table_pdf_fixtures,convert_sec_html_to_pdf}.py` | untracked | fixture tooling | review with fixture rights | Generates/transforms fixtures |
| `scripts/{build_regression_comparison,finalize_regression_readiness,rescore_e2e_production_audit,validate_p0_deep,validate_p0_optimizations}.py` | untracked | obsolete temporary / uncertain | do not stage or delete | Phase-specific one-off audit helpers |
| `scripts/check_{deepseek,dashscope_embedding}.py` | modified + untracked | diagnostics | stage after review | Redacted credential/provider diagnostics |
| `.vscode/settings.json` | modified | uncertain local/editor config | do not stage or delete | Only exposes `outputs/` in local explorer |
| `fixtures/stress/*.pdf` | untracked | synthetic test fixture | stage after confirming generated ownership | Small generated table fixtures (4.9–8.8 KB) |
| `fixtures/e2e_real/*.htm`, `*.pdf` | untracked | uncertain third-party-derived fixture | do not stage or delete | SEC issuer filing redistribution rights need owner review |
| Local root `LumenFin_*.md` reports | ignored/untracked | generated historical reports | preserve locally; do not stage | Superseded by canonical `reports/` summaries |
| `outputs/`, `test_artifacts/` | ignored | generated outputs | keep ignored | Test/live products |
| `data/*.db`, Milvus Lite directories, `*.db` | ignored | local runtime state | keep ignored | SQLite/vector state and locks |
| `.venv/`, `*.egg-info`, `__pycache__`, `.pytest_cache` | ignored | cache/build state | keep ignored | Machine-specific artifacts |
| `.env` | ignored | secrets | prohibit commit | Real local credentials |
| `logs/`, `*.log` | ignored | generated logs | keep ignored | Runtime diagnostics |

## Binary and large-file review

Untracked real-fixture binaries:

| File | Size |
|------|-----:|
| `aapl_fy2024_10k_sec.pdf` | 99,891 B |
| `aapl_fy2024_10k_sec_long.pdf` | 128,513 B |
| `msft_fy2024_10k_sec.pdf` | 67,581 B |
| `msft_fy2024_10k_sec_long.pdf` | 207,382 B |
| `nvda_fy2025_10k_sec.pdf` | 118,117 B |
| `tsla_fy2024_10k_sec.pdf` | 113,611 B |

The files are not unusually large for Git, but size is not the blocker:
redistribution/provenance is uncertain. `msft-20240630.htm` is also a
substantial 25,069-line filing source. None is staged.

Synthetic stress PDFs are under 9 KB each and remain pending ownership
confirmation.

## Secret and path review

- No tracked literal API key/token pattern was found.
- `.env` and local databases are ignored and absent from `git ls-files`.
- Canonical release docs use relative paths/environment discovery.
- Local historical reports and ignored interview notes may still contain
  workstation paths but are excluded from release input.

## Uncertain items requiring human decision

1. SEC-derived HTML/PDF fixture redistribution.
2. Whether phase-specific one-off validation scripts belong in the public RC.
3. `.vscode/settings.json` local explorer preference.
4. PyMuPDF AGPL/commercial implications for distributed Docker images.

These items are preserved, unstaged and undeleted.

## Release conclusion

Production code, tests, CI and formal docs can be staged explicitly by commit
group. The tree cannot become clean until uncertain items are either approved
for release, intentionally ignored with documented consequences, or otherwise
resolved by the owner.
