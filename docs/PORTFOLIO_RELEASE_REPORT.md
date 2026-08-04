# LumenFin Portfolio Release Report

Release candidate freeze evidence for interview / portfolio use. Every number
below comes from a recorded run in this repository; nothing here is projected.

## 1. Release candidate

| Field | Value |
|-------|-------|
| Name | **LumenFin Portfolio Release Candidate** |
| Version | `0.1.0rc2` (recommended tag `v0.1.0-rc.2`) |
| Positioning | Portfolio release candidate / controlled deployment candidate |
| FinRun schema | `1.0` |
| FinAgentBench pin | `v0.1.0-rc.1` |

> LumenFin is a portfolio release candidate validated under controlled
> multi-process and deterministic fault-injection conditions. These results are
> not a certification of unrestricted production readiness.

## 2. Release commit / worktree / runtime

| Field | Value |
|-------|-------|
| Baseline commit | `0fa91f60977f55d9a7605bedf7e6f79c97f25f86` |
| Phase 4.0 commits | `40fe781` docs(architecture) → `74264b1` docs(readme) → `cfb7443` ci → `659c725` feat(demo) → `8897aeb` fix(test) → this release-docs commit |
| Release commit | current `main` HEAD after the commit series above (tag target) |
| Worktree at freeze | clean (verified before tagging) |
| Local Python | 3.12 (Windows) |
| CI Python | 3.12 (`ubuntu-latest`) |
| Exactly-once | **not claimed** |

## 3. Agent / evidence architecture

```text
Query / PDF
→ LangGraph orchestration
→ SEC / Yahoo / hybrid RAG
→ structured financial grounding
→ claim builder
→ evidence binder
→ verified-only synthesis
→ FinRun
→ FinAgentBench
```

Only claims bound to evidence reach synthesis. Missing fundamentals produce
`incomplete_data` instead of forged numerics.

## 4. Runtime architecture

```text
Load balancer / direct API ports
        ↓
API A / API B
        ↓
PostgreSQL  (checkpoints, jobs, RAG documents, canonical chunks, leases/attempts)
        ↓
Redis       (pending / processing / dead-letter)
        ↓
Index Worker A / B
        ↓
Milvus Server
```

## 5. Store responsibilities

| Store | Owns |
|-------|------|
| PostgreSQL | Source of truth: checkpoints (CAS), analysis jobs, RAG document/chunk metadata, index leases and attempt counters |
| Redis | Reliable work queues only: `pending` → `processing` → dead-letter; reclaim of idle processing entries |
| Milvus Server | Vector plane; rows keyed and filtered by `tenant_id` |

## 6. Reliable queue semantics

- At-least-once delivery (never exactly-once).
- `pending → processing` handoff is atomic; abandoned processing entries are
  reclaimed by idle timeout — **no manual redelivery** after a worker kill.
- Attempt fencing: index leases carry an attempt number; stale workers cannot
  overwrite a newer attempt.
- Exhausted attempts land in the dead-letter queue instead of silent loss.
- Checkpoint writes use compare-and-set so two API processes cannot clobber
  each other.

## 7. Provider resilience semantics

Request-level deadline; bounded max attempts; `Retry-After` honored;
exponential backoff with jitter; a **single retry owner** per logical call (no
nested retry loops); process-local shared HTTP client; explicit `degraded`
marker on fallback; per-process bulkhead (concurrency cap) per provider class;
deterministic provider stub for reproducible fault injection.

## 8. Multi-tenant isolation scope

Scope: **RAG data-plane tenant-aware logical isolation**.

Covered: canonical document ID = `tenant_id + content_hash`; tenant-filtered
PostgreSQL RAG CRUD; `tenant_id` carried in Redis index payloads; Milvus row key
and metadata carry `tenant_id` with filter push-down; keyword and vector
retrieval both tenant scoped; measured integration leakage `0`.

Not covered: tenant identity bound to credentials, per-tenant API keys / JWT
claims, fully tenant-scoped checkpoints and analysis jobs, PostgreSQL RLS,
per-tenant databases/collections. Details:
[MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md).

## 9. Validated gates

Kept separate on purpose — these are four different questions.

| Gate | Question | Run id | Result |
|------|----------|--------|--------|
| Unit regression | Does offline logic still hold? | this release | **453 passed, 1 skipped** |
| Infrastructure integration (Phase 3.2B) | Do multi-process queue/worker/DB semantics hold under kill? | `20260804T095357Z` | **PASS** |
| Provider fault validation (Phase 3.3A) | Do deadlines/retries/bulkheads hold under injected faults across two API containers? | `docker_20260804T100817Z` | **PASS** |
| Benchmark reliability (FinAgentBench) | Are exported runs judged reliable by an external gate? | pin `v0.1.0-rc.1` | completed-case mean **92.97**, mutation **4/4** |

Phase 3.2B detail: worker-kill manual redelivery `false`; tenant leakage `0`;
orphan chunks / vectors `0 / 0`.

Phase 3.3A Docker detail: provider logical calls `20`; physical attempts `25`;
cross-container context leakage `0`; unexpected failures `0`.

Phase 3.2B and 3.3A were **not re-run** in Phase 4.0: this phase changed only
documentation, CI workflows, and the demo entrypoint — no queue, worker,
database, checkpoint, Milvus filtering, provider resilience, or Compose shared
infrastructure code was modified.

## 10. CI

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `.github/workflows/ci.yml` | push, pull_request | Offline regression (`run_tests.py`) + doc link check + portfolio demo, Python 3.12, lockfile, no API keys, no live providers |
| `.github/workflows/integration-manual.yml` | `workflow_dispatch` | Documents Docker-capable Phase 3.2B / 3.3A commands; not a per-commit gate |
| `.github/workflows/test.yml` | `workflow_dispatch` | Extended cross-repo FinAgentBench gate (was per-push; demoted so cross-repo checkout does not block every commit) |

Local equivalents of every `ci.yml` step were executed on Windows before the
freeze and all passed.

Observed GitHub run: workflow `ci`, run `30905055570`, commit
`2839c1c223a2be6589c6b349d270129a74c37dc2` — **success**. The only commit after
that point records this CI result in this report; no code, test, workflow, or
demo behavior changed after the verified run.

## 11. Demo

```powershell
python scripts/run_portfolio_demo.py
```

Offline, no API key, no live provider access, writes only under
`outputs/portfolio_demo/`, prints a JSON summary, exits non-zero on failure.

| Demo | Observed |
|------|----------|
| A trusted analysis | `completed`, Apple-only scope, 5 claims, evaluator score 100 |
| B isolation + mutations | `completed`, companies `["Apple","Microsoft"]`, mutation checks 4/4 |
| C fail-closed | `incomplete_data`, 0 numeric claims, provider/data absence distinguished |

Optional (not started by the demo): Docker recovery story
`worker A killed → automatic reclaim → worker B attempt=2 → ready`
([Phase 3.2B](PHASE32B_INTEGRATION_REPORT.md)).

## 12. Known limitations

- At-least-once queue delivery, **not** exactly-once
- Per-process bulkheads, **not** a cross-process global rate limit
- No shared circuit breaker across API processes
- Live provider smoke: **skipped** in this release evidence
- No large-scale soak / long-duration endurance run
- Tenant identity is not bound to authentication (no IAM tenant binding)
- No PostgreSQL Row-Level Security
- PyMuPDF (AGPL) constrains redistribution of derived artifacts
- Output is research support, **not investment advice**

## 13. Live smoke status

`skipped`. No live DeepSeek / DashScope / SEC / Yahoo calls were made for this
release evidence pack. All validated numbers come from offline tests,
deterministic stubs, or containerized infrastructure.

## 14. License boundary

Code is portfolio/demo licensed as declared in the repository. PyMuPDF is AGPL —
redistributing rendered filing images publicly is out of scope. SEC and Yahoo
data usage follows each provider's terms; `SEC_USER_AGENT` must be operator
owned in any real deployment.

## 15. Remaining roadmap (not in this release)

1. Bind `tenant_id` to authenticated identity (API keys / JWT claims)
2. Tenant-scope checkpoints and analysis jobs; add PostgreSQL RLS
3. Cross-process global rate limiting and a shared circuit breaker
4. Live provider smoke in a controlled quota window
5. Multi-hour soak with fault injection
6. Managed vector infrastructure and horizontal worker autoscaling

## 16. Recommended tag

```powershell
git tag -a v0.1.0-rc.2 -m "LumenFin portfolio release candidate v0.1.0-rc.2"
git push origin v0.1.0-rc.2
```

Create the tag only after human review of this report.
