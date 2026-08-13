# LumenFin Portfolio Release Report

Release candidate freeze evidence for portfolio / controlled-demo use. Every
number below comes from a recorded run in this repository; nothing here is
projected.

> **Immutable snapshot boundary:** this report describes tag
> `v0.1.0-rc.3` at commit `e67ed5f0e5aa4d2585d063b99212c46f5002d1a3`.
> Post-tag `main` added principal-bound tenant authorization and tenant-scoped
> jobs/checkpoints; current scope is documented in
> [MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md). Historical statements
> and counts below are intentionally retained as tag evidence.

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

## 8. Multi-tenant isolation scope at the RC tag

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
| RC-tag LumenFin full regression | Does the frozen Linux image pass the complete suite? | Phase 6 tag worktree | **495 passed, 2 skipped** |
| RC-tag FinAgentBench full regression | Does the frozen evaluator pass its complete suite? | Phase 6 tag worktree | **149 passed** |
| Infrastructure integration (queue/worker) | Do multi-process queue/worker/DB semantics hold under kill? | `20260804T095357Z` | **PASS** |
| Provider fault validation (provider resilience) | Do deadlines/retries/bulkheads hold under injected faults across two API containers? | `docker_20260804T100817Z` | **PASS** |
| Benchmark reliability (FinAgentBench) | Are exported runs judged reliable by an external gate? | informational pin `v0.1.0-rc.1` | completed-case mean **92.97**, mutation **4/4** (not a score for current pin `v0.1.0-rc.3`) |
| Evaluator compatibility (FinAgentBench) | Does the frozen FinRun export replay on the current pin? | pin `v0.1.0-rc.3` | **PASS** (schema `1.0`; core 4/4 + extended 7/7 on compatibility gate) |
| Native BM25 cutover | Does dense + native BM25 weighted RRF pass offline and live first-search gates? | 2026-08-12 local closure | **PASS** |
| Qwen3 rerank | Do hard-negative quality and telemetry gates pass with zero fallback? | 2026-08-12 synthetic live gate | **PASS** |
| Cross-repository contract | Does current FinRun `1.0` pass the evaluator and negative controls? | Phase 6 current worktrees | score **100**, mutations **11/11** |
| Production hardening | Do immutable builds, UID 10001, readiness, persistence, fallback, backup, leak, and graceful-stop gates pass? | 2026-08-12 Phase 6 | **PASS** |

Queue/worker detail: worker-kill manual redelivery `false`; tenant leakage `0`;
orphan chunks / vectors `0 / 0`.

Provider-resilience Docker detail: provider logical calls `20`; physical attempts `25`;
cross-container context leakage `0`; unexpected failures `0`.

The full validation evidence, including harness corrections and the dirty-worktree
boundary, is frozen in
[`PRODUCTION_LIMITATIONS.md`](../docs/PRODUCTION_LIMITATIONS.md).

## 10. CI

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `.github/workflows/ci.yml` | push, pull_request | Offline regression (`run_tests.py`) + doc link check + portfolio demo, Python 3.12, lockfile, no API keys, no live providers |
| `.github/workflows/integration-manual.yml` | `workflow_dispatch` | Documents Docker-capable queue/worker / provider-resilience commands; not a per-commit gate |
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
([queue/worker](QUEUE_WORKER_INTEGRATION.md)).

## 12. Known limitations at the RC tag

- At-least-once queue delivery, **not** exactly-once
- Per-process bulkheads, **not** a cross-process global rate limit
- No shared circuit breaker across API processes
- Controlled synthetic DeepSeek, DashScope embedding, and Qwen3 rerank smoke
  passed; tag `v0.1.0-rc.3` and remote CI are published/green. Remaining gaps
  are soak, IAM-bound tenancy, and public image redistribution
- No large-scale soak / long-duration endurance run
- At the tag freeze, tenant identity was not bound to authentication
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
It did not run a fresh live SEC/Yahoo matrix. Published RC tag `v0.1.0-rc.3`
and green remote CI close the former Phase 7 commit/tag/release boundary;
remaining roadmap items are in §15.

## 14. License boundary

Project-owned source is MIT licensed, copyright 2026 Jiali Ma. That grant does
not relicense dependencies or data. PyMuPDF and MinIO are AGPL-3.0 (or
commercial terms where offered), and Redis 7.4 is RSALv2/SSPLv1. The current
application image must not be represented as a purely MIT distributable.
Details: `THIRD_PARTY_NOTICES.md`.

SEC and Yahoo data usage follows each provider's terms; `SEC_USER_AGENT` must
be operator owned in any real deployment.

## 15. Roadmap recorded at the tag freeze

1. Bind `tenant_id` to authenticated identity (API keys / JWT claims)
2. Tenant-scope checkpoints and analysis jobs; add PostgreSQL RLS
3. Cross-process global rate limiting and a shared circuit breaker
4. Live provider smoke in a controlled quota window
5. Multi-hour soak with fault injection
6. Managed vector infrastructure and horizontal worker autoscaling

## 16. Tag boundary

Published immutable tag: **`v0.1.0-rc.3`**
(`e67ed5f0e5aa4d2585d063b99212c46f5002d1a3`). Subsequent `main` docs commits
may land ahead of that tag without changing the RC code freeze. Do not move or
reuse the tag; cut a new RC if another immutable snapshot is required.
