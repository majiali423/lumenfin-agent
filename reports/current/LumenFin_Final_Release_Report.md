# LumenFin Final Release Report

Candidate: `0.1.0rc1`
Suggested tag: `v0.1.0-rc.1`
FinRun: `1.0`
Assessment date: 2026-07-25

## Positioning

LumenFin is a **Trustworthy Financial Research Agent** for controlled
deployment. It is not an unrestricted production/HA or automated investment
system.

## Release changes

- SEC 10-K/live issuer financial grounding
- issuer isolation and request-scoped runtime
- claim → evidence binding
- fail-closed incomplete-data paths
- FinRun `1.0` export
- FinAgentBench tag contract and CI gate
- locked Python dependencies
- portable path/configuration documentation
- production limitations and demo guide

No Agent, claim rule, evaluator metric or threshold was added during release
finalization.

## Validation

| Gate | Result |
|------|--------|
| LumenFin offline tests | 265 PASS, 1 skipped |
| FinAgentBench tests | 75 PASS |
| Mutation gate | 4/4 |
| Correctness validation | PASS |
| Cross-repository FinRun gate | PASS |
| Offline demo | PASS |
| Live RC | 8/8 PASS |
| Completed-case FAB mean | 92.97 |
| Linter | No new diagnostics |
| Docker build | NOT VERIFIED — local Docker daemon unavailable |

## Security/configuration audit

| Finding | File/scope | Severity | Action |
|---------|------------|----------|--------|
| No committed API key/token pattern found | tracked release candidates | PASS | Keep `.env` ignored |
| SEC operator identity previously had dev fallback in all modes | `sec_fundamentals.py` | High | Production now requires `SEC_USER_AGENT`; regression tests added |
| Compose previously inherited dev/demo defaults and example credentials | `docker-compose.yml` | Blocker | Forces production/live, requires secrets, removes database port publishing |
| API/OpenAPI version diverged from package version | `api/app.py` | High | Reads installed package metadata; regression test added |
| Local absolute paths existed in linked docs/scripts | cross-repo scripts/docs | High | Replaced with sibling/env discovery |
| Generated reports/probes cluttered repository root | local working tree | Medium | Canonical reports moved to `reports/`; local generated files ignored |
| Repository owner was explicit in CI checkout | workflow | Low | Uses `github.repository_owner` |

## Open release blockers

1. The working tree contains a large set of reviewed-but-uncommitted RC changes.
2. FinAgentBench tag `v0.1.0-rc.1` does not yet exist; LumenFin CI intentionally
   pins that tag and will fail until the benchmark is published first.
3. Docker image build could not be validated because Docker Desktop/Linux
   engine was not running.
4. No repository license has been selected; public distribution requires an
   explicit owner decision on license and third-party notices.

## Recommendation

Engineering behavior is suitable for a **Release Candidate** after blockers
above are closed. Stable release is not recommended because managed vector,
HA checkpoint, load/soak and provider SLO validation remain out of scope.
