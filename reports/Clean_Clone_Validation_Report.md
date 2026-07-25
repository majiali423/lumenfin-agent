# Clean Clone Validation Report

Date: 2026-07-26  
Method: detached git worktrees from local candidate HEADs (no `.env` / outputs / DB copy)

## Candidate HEADs

| Repository | HEAD |
|------------|------|
| FinAgentBench | `e5cffe2e9fca3d7c052b2c5db193d0f6796008f6` |
| LumenFin | `8b944f7d9927bc2a726fc57ec5b12f3c4b7ebd10` (includes fixture-path follow-up) |

Initial dual worktree run used LumenFin `f503f68`; fixture-path commit `8b944f7` was added afterward so optional SEC tests no longer depend on ignored `fixtures/e2e_real/`.

## Results

| Gate | Result |
|------|--------|
| FAB `unittest discover` | 77 tests OK |
| FAB mutation suite | 4/4 |
| FAB offline demo | PASS |
| FAB import side-effect tests | 2/2 PASS |
| LumenFin `scripts/run_tests.py` | 267 tests OK (1 skip: live integration unless `RUN_INTEGRATION_TESTS=1`) |
| Cross-repo gate (`ci`) | PASS; both worktrees dirty=false |
| RC `--dry-run` | PASS |

## Notes

- Offline demo / cross-repo / RC dry-run entrypoints live in FinAgentBench.
- Clean worktrees did not copy `.env`, databases, Milvus state, or full SEC downloads.
- Remaining untracked in developer trees (not required for clean clone): inventory/cleanup plans, portfolio notes, generated stress PDFs.

## Judgment for Phase 6

Offline clean-clone gates: **PASS**. Live RC may proceed when live providers are configured.
