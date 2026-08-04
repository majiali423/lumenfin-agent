# Changelog

## 0.1.0rc2 — 2026-08-04

Portfolio release candidate validated under controlled multi-process and
deterministic fault-injection conditions. Not a certification of unrestricted
production readiness.

### Added

- PostgreSQL-first runtime: checkpoints (compare-and-set), analysis jobs, RAG
  document/chunk metadata, index leases and attempt counters
- Redis reliable queue: `pending → processing → dead-letter` with idle reclaim
  (worker kill recovers without manual redelivery)
- Multi-process recovery: index lease + attempt fencing across workers
- Provider resilience layer: request deadline, bounded attempts, `Retry-After`,
  exponential backoff with jitter, single retry owner, process-local shared HTTP
  client, degraded fallback marker, per-process bulkheads, deterministic stub
- Docker dual-API validation (two API containers + two index workers + Milvus
  Server) with cross-container context-leakage checks
- RAG data-plane tenant isolation evidence (tenant-scoped IDs, PostgreSQL
  filters, Redis payload `tenant_id`, Milvus filter push-down)
- `docs/MULTI_TENANCY_BOUNDARY.md` — covered vs not-covered threat table
- `docs/PORTFOLIO_RELEASE_REPORT.md` — release freeze evidence
- `scripts/run_portfolio_demo.py` — deterministic offline A/B/C demo entrypoint
- `.github/workflows/ci.yml` — offline regression + demo on push/PR
- `.github/workflows/integration-manual.yml` — manual Docker suite instructions

### Changed

- README rebuilt around two architecture paths (agent/evidence and
  runtime/infrastructure) with per-gate validated results
- Production positioning unified: portfolio release candidate / controlled
  deployment candidate (no conflicting "production ready" vs "not production
  ready" claims)
- Architecture docs describe Milvus Server for multi-process runs, not only
  Milvus Lite
- `.github/workflows/test.yml` demoted to `workflow_dispatch` so the cross-repo
  FinAgentBench checkout no longer gates every commit

### Fixed

- `test_concurrent_same_tenant_same_content_has_one_canonical_index` no longer
  asserts a scheduling-dependent loser status; it asserts the real invariant
  (one ready receipt, one canonical document, one vector index call)

### Validated

| Gate | Result |
|------|--------|
| Offline regression | 453 passed, 1 skipped |
| Phase 3.2B integration (`20260804T095357Z`) | PASS — worker-kill manual redelivery false, tenant leakage 0, orphan chunks/vectors 0/0 |
| Phase 3.3A Docker dual-API (`docker_20260804T100817Z`) | PASS — logical 20 / physical 25, context leakage 0, unexpected failures 0 |
| FinAgentBench (pin `v0.1.0-rc.1`) | completed-case mean 92.97, mutation 4/4 |
| Live provider smoke | skipped |

### Known limitations

- At-least-once delivery, not exactly-once
- Per-process bulkheads, not cross-process global rate limiting
- No shared circuit breaker; no large-scale soak
- Tenant identity not bound to authentication; no PostgreSQL RLS
- PyMuPDF (AGPL) limits redistribution of derived filing images
- Research support only; not investment advice

## 0.1.0rc1 — 2026-07-25

Release candidate for controlled production deployment.

### Added

- Issuer-only SEC/Yahoo financial grounding
- Claim → evidence binding before report synthesis
- Fail-closed reporting for unavailable/sparse fundamentals
- Request-scoped Agent runtime with concurrent issuer isolation test
- FinRun schema `1.0` export and pinned FinAgentBench release contract
- Locked dependencies and portable cross-repository validation

### Validated

- LumenFin full offline regression
- FinAgentBench correctness and four-mutation gates
- Eight-case live RC pack (issuer, long document, compare and fail-closed)

### Security / release

- Production SEC access requires an operator-owned `SEC_USER_AGENT`
- Secrets, local databases, outputs and caches are excluded from release input

### Known limitations

- Controlled deployment only; Milvus Lite and SQLite are not HA infrastructure
- Live behavior depends on external LLM, embedding, SEC and market providers
- Live structured-source citations do not imply filing page citations
