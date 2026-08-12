# LumenFin Portfolio Release Report

Release candidate freeze evidence for portfolio / controlled-demo use. Every
number below comes from a recorded run in this repository; nothing here is
projected.

## 1. Release candidate

| Field | Value |
|-------|-------|
| Name | **LumenFin Portfolio Release Candidate** |
| Version | `0.1.0rc3` / published tag `v0.1.0-rc.3` |
| Positioning | Controlled release candidate / portfolio demo candidate |
| FinRun schema | `1.0` |
| FinAgentBench pin | `v0.1.0-rc.3` (evaluator); current FinAgentBench package tag `v0.1.0-rc.4` |

> LumenFin is a portfolio release candidate validated under controlled
> multi-process and deterministic fault-injection conditions. These results are
> not a certification of unrestricted production readiness.

## 2. Release closure state / runtime

| Field | Value |
|-------|-------|
| Baseline branch | `main` |
| Release commit | `e67ed5f0e5aa4d2585d063b99212c46f5002d1a3` |
| Worktree at freeze | Clean at tag `v0.1.0-rc.3` |
| Tag | `v0.1.0-rc.3` (published) |
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
| Milvus Server | `lumenfin_chunks_v4_bm25`: 1024-D dense vectors + native BM25 sparse function, tenant-filtered; weighted RRF candidates feed Qwen3 rerank |

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

Kept separate on purpose — these gates answer different questions.

| Gate | Question | Run id | Result |
|------|----------|--------|--------|
| Current LumenFin full regression | Does the final Linux image pass the complete suite? | Phase 6 current worktree | **495 passed, 2 skipped** |
| Current FinAgentBench full regression | Does the evaluator pass its complete suite? | Phase 6 current worktree | **149 passed** |
| Infrastructure integration (Phase 3.2B) | Do multi-process queue/worker/DB semantics hold under kill? | `20260804T095357Z` | **PASS** |
| Provider fault validation (Phase 3.3A) | Do deadlines/retries/bulkheads hold under injected faults across two API containers? | `docker_20260804T100817Z` | **PASS** |
| Benchmark reliability (FinAgentBench) | Are exported runs judged reliable by an external gate? | pin `v0.1.0-rc.3` | completed-case mean **92.97**, mutation **4/4** |
| Native BM25 cutover | Does dense + native BM25 weighted RRF pass offline and live first-search gates? | 2026-08-12 local closure | **PASS** |
| Qwen3 rerank | Do hard-negative quality and telemetry gates pass with zero fallback? | 2026-08-12 synthetic live gate | **PASS** |
| Cross-repository contract | Does current FinRun `1.0` pass the evaluator and negative controls? | Phase 6 current worktrees | score **100**, mutations **11/11** |
| Production hardening | Do immutable builds, UID 10001, readiness, persistence, fallback, backup, leak, and graceful-stop gates pass? | 2026-08-12 Phase 6 | **PASS** |

Phase 3.2B detail: worker-kill manual redelivery `false`; tenant leakage `0`;
orphan chunks / vectors `0 / 0`.

Phase 3.3A Docker detail: provider logical calls `20`; physical attempts `25`;
cross-container context leakage `0`; unexpected failures `0`.

The full Phase 6 evidence, including harness corrections and the dirty-worktree
boundary, is frozen in
[`PHASE6_FULL_VALIDATION_REPORT.md`](../reports/current/PHASE6_FULL_VALIDATION_REPORT.md).

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
- Controlled synthetic DeepSeek, DashScope embedding, and Qwen3 rerank smoke
  passed; clean committed/tagged RC validation remains a Phase 7 boundary
- No large-scale soak / long-duration endurance run
- Tenant identity is not bound to authentication (no IAM tenant binding)
- No PostgreSQL Row-Level Security
- PyMuPDF (AGPL) constrains redistribution of derived artifacts
- Output is research support, **not investment advice**

## 13. Live smoke status

On 2026-08-12, controlled synthetic data passed the production
API → PostgreSQL → DashScope embedding → Milvus dense/native-BM25 → Qwen3
rerank path with zero rerank fallback/degradation. A synthetic upload also
completed a DeepSeek analysis. No user document was sent externally.

Phase 6 also completed both repositories' full suites, the FinRun contract
gate, image/runtime checks, backup verification, leak scan, and graceful stop.
It did not run a fresh live SEC/Yahoo matrix; clean committed/tagged RC
validation remains a Phase 7 boundary.

## 14. License boundary

Project-owned source is MIT licensed, copyright 2026 Jiali Ma. That grant does
not relicense dependencies or data. PyMuPDF and MinIO are AGPL-3.0 (or
commercial terms where offered), and Redis 7.4 is RSALv2/SSPLv1. The current
application image must not be represented as a purely MIT distributable.
Details: `THIRD_PARTY_NOTICES.md`.

SEC and Yahoo data usage follows each provider's terms; `SEC_USER_AGENT` must
be operator owned in any real deployment.

## 15. Remaining roadmap (not in this release)

1. Bind `tenant_id` to authenticated identity (API keys / JWT claims)
2. Tenant-scope checkpoints and analysis jobs; add PostgreSQL RLS
3. Cross-process global rate limiting and a shared circuit breaker
4. Live provider smoke in a controlled quota window
5. Multi-hour soak with fault injection
6. Managed vector infrastructure and horizontal worker autoscaling

## 16. Tag boundary

No tag is created by this report. Phase 7 must inspect existing local and
remote tags, choose a fresh immutable RC version if required, update package and
release metadata consistently, and obtain explicit approval before commit,
push, tag, or release creation.
