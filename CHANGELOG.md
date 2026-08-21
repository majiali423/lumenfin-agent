# Changelog

## 0.1.0rc5 — candidate preparation (unpublished)

This section records source-candidate identity for current `main`. It is **not**
a published release: no `v0.1.0-rc.5` tag, GitHub Release, or public Docker
image has been created. Latest published LumenFin remains
`0.1.0rc4` / `v0.1.0-rc.4`.

### Identity

- Python package, API fallback version, and Compose default image tag:
  `0.1.0rc5`
- Intended annotated tag name (not created in this preparation): `v0.1.0-rc.5`

### Documentation

- Relabeled the 2026-08-13 Linux-image suite (**512 passed, 3 skipped**) as a
  dated post-rc4 snapshot. It is not an undated “current main” count.

### Evaluation

- Sealed the LEDGER `public_dev` retrieval / packing / generate canary chain
  under `data/eval_rag/holdout/`. Parent-page *return* stays eval-only; do not
  embed a page-parent index. `public_holdout` remains unopened. FinanceBench
  Phase 4 stays `NOT_RUN`. Production RAG defaults are unchanged.
- Added `data/eval_rag/holdout/ledger_public_dev_chain_seal.json` so sealed
  public-dev artifacts can be hash-checked as one provenance chain. Recommended
  annotated tag name `ledger-public-dev-chain-v1` is recorded but not created.

### CI

- Required FinRun contract gate now runs two fail-closed lanes: FinAgentBench
  `v0.1.0-rc.3` (authoritative frozen pin) and published `v0.1.0-rc.4`
  (latest-release compatibility). Artifact names include the lane and ref.
  FinAgentBench `master` is not a required gate.

### Contract

- Added structured answer citation schema `1.0` so verified `chunk_id` values
  can be read by API, FinRun, and LEDGER without guessing IDs from prose.
  This is a contract, not an accuracy claim. `public_holdout` stays closed.
- Fail-closed citation boundaries: illegal IDs degrade with explicit
  `citation_validation=failed`; `citation_source` write values are
  `structured` / `legacy_structured` / `unavailable`.
- Added an offline synthetic structured-citation end-to-end canary
  (`scripts/run_structured_citation_canary.py`). It is a contract check,
  not product accuracy, FinanceBench, or a LEDGER score. `public_holdout`
  stays closed.

## 0.1.0rc4 — 2026-08-13

Controlled release candidate for tenant authorization, upgrade safety, and
auditable Qwen3 evidence. The annotated tag `v0.1.0-rc.4` is published at
commit `90f9e94b7b7af7bc61ee35ee56cec1bdb56ccf55`.

### Fixed

- Added PostgreSQL migration `003` to the automatic migrator and validated the
  pre-003 schema upgrade path with preserved legacy ownership
- Wired production API-key principal maps through Compose while retaining
  fail-closed startup and legacy single-key compatibility
- Removed raw API keys from malformed-principal startup errors

### Evidence and documentation

- Aligned tenant scope and claim-module paths across current documentation
- Preserved the synthetic Qwen3 live gate as sanitized machine-readable
  evidence with hash, privacy, metric, and telemetry checks
- Kept immutable `v0.1.0-rc.3` evidence separate from post-tag `main` results

### Validation to date

- Current Linux-image offline suite: 512 passed, 3 skipped
- PostgreSQL pre-003 upgrade gate: PASS
- Production multi-principal Compose rendering: PASS
- Production-stack migration, health, authentication, and synthetic RAG
  end-to-end gate: PASS
- GitHub Actions on `main` and `v0.1.0-rc.4`: PASS

The [GitHub Pre-release](https://github.com/majiali423/lumenfin-agent/releases/tag/v0.1.0-rc.4)
and immutable Git tag are published.

## 0.1.0rc3 — 2026-08-12

Controlled portfolio RC closure with Milvus 3, native BM25, optional Qwen3
reranking, production-runtime hardening, backup verification, and MIT licensing
for project-owned source.

### Added

- Versioned `lumenfin_chunks_v4_bm25` dense/native-BM25 collection and rebuild
  tooling
- DashScope Qwen3 rerank with bounded retry, telemetry, lexical fallback, and
  hard-negative quality gates
- Deep readiness, persistent Redis AOF, verified production backup, and
  graceful worker shutdown
- MIT `LICENSE`, expanded third-party notices, and Phase 6 full validation
  evidence

### Validation

- LumenFin full Linux-image suite: 495 passed, 2 skipped
- FinAgentBench full suite: 149 passed
- FinRun `1.0` cross-repository gate: score 100; negative controls 11/11

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
