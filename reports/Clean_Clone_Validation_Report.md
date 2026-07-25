# Clean Clone Validation Report

Date: 2026-07-26
Method: detached git worktrees from final local HEADs (no `.env` / outputs / DB copy from developer trees)

## Candidate HEADs (final post-RC commits)

| Repository | HEAD | Subject |
|------------|------|---------|
| LumenFin | `0f895f85fdd3c39446900c639caaa616d9e7a756` | `fix(config): fail fast on conflicting provider credentials` |
| FinAgentBench | `f58e47978af1badf431221bb1911c0c952b982f1` | `fix(validation): reject fallback during live RC` |

Worktrees: clean (`status --porcelain` empty) at validation start.

## Results

| Gate | Result |
|------|--------|
| LumenFin `tests.test_env_bootstrap_conflicts` | 4/4 OK |
| LumenFin `scripts/run_tests.py` | **271** OK (1 skip: live integration unless `RUN_INTEGRATION_TESTS=1`) |
| FAB `unittest discover` | **78** OK (1 skip) |
| FAB `tests.test_rc_runner_import` | 3/3 OK |
| FAB mutation suite | **4/4** |
| FAB offline demo | PASS |
| Cross-repo gate (`ci`) | PASS; both worktrees dirty=false |
| RC `--dry-run` | PASS |

## Live RC consistency

Live 8-case pack ran earlier on HEADs `2e28d74` / `6700846` with process env cleaned (`LIVE_RC_EXIT=0`).
Delta to these final HEADs is env fail-fast + live-RC fallback abort + reports only — no Agent/case/fixture/threshold changes. Full live pack not re-run.

## Judgment

Final-HEAD offline clean-clone gates: **PASS**.
