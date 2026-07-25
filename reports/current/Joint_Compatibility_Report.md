# LumenFin ↔ FinAgentBench Compatibility Report

Assessment date: 2026-07-25

## Contract

| Producer | Producer version | FinRun | Evaluator | Supported |
|----------|------------------|--------|-----------|:---------:|
| LumenFin | `0.1.0rc1` | `1.0` | FinAgentBench `0.1.0rc1` | YES |

Recommended release tags:

- FinAgentBench: `v0.1.0-rc.1`
- LumenFin: `v0.1.0-rc.1` (published second)

## Gate evidence

The release cross-repository gate recorded:

- LumenFin commit: `f13ec3d867fa53de0594a6a8c992e9c2ba1e6f6f`
- FinAgentBench commit: `a2042e6a493af1d5e464590eeb082bec7c20fa70`
- Both worktrees dirty: **yes**
- FinRun schema: `1.0`
- Profile: `ci`
- FinAgentBench gate: PASS
- Mutation gate: PASS (`4/4`, detection rate `1.0`)

The commit hashes identify the current HEADs, while the dirty flags explicitly
show that release changes are not represented by those commits yet. Final
published compatibility evidence must be regenerated from clean tagged trees.

## Compatibility policy

1. LumenFin CI defaults to the released FinAgentBench tag, never a floating
   branch.
2. Manual workflow dispatch may override the tag with a reviewed SHA/ref.
3. The resolved benchmark SHA is printed by CI.
4. Unknown FinRun schema versions fail before scoring.
5. Legacy unversioned FinRuns are transition-only.
6. Compatibility never lowers benchmark thresholds.

## Release order

1. Commit/test/tag FinAgentBench.
2. Confirm tag availability and CI artifacts.
3. Commit LumenFin with `FINAGENTBENCH_REF=v0.1.0-rc.1`.
4. Run offline + live RC from clean trees.
5. Tag LumenFin.
