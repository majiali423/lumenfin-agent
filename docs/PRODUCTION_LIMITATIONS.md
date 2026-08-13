# Production Boundaries and Limitations

LumenFin is a **portfolio release candidate** validated under controlled
multi-process and deterministic fault-injection conditions. These results
are **not** a certification of unrestricted production readiness.

| Status | Meaning |
|--------|---------|
| Controlled environment acceptance | Queue/worker integration, provider-resilience, and offline regression PASS |
| Suitable for | Internal demos, portfolio showcases, limited operator-owned deploys |
| Not claimed | Internet-scale production certification, exactly-once delivery |

## Current suitable scope

- Controlled internal / demo deployment
- One or few API instances + index workers with explicit process ownership
- Services with provider quotas, monitoring, and operator response
- Financial research support with human review
- Replayable, audited Agent runs

## Current non-goals

- Internet-scale multi-tenant high availability
- Strongly consistent distributed checkpoints across regions
- Managed Milvus cluster autoscaling
- Zero-downtime model-provider switching
- High-frequency trading or trade execution
- Automated investment, legal, or fiduciary decisions
- Full SaaS IAM / per-tenant billing

## Known limitations

### Delivery and concurrency

- Queue semantics are **at-least-once**, not exactly-once
- Reserve moves pending→processing with a short poll sleep (Lua atomic move);
  not BRPOP-style blocking waits
- Bulkheads are **per-process**, not a cross-process global rate limit
- No shared circuit breaker across API processes
- Controlled synthetic DeepSeek/DashScope/Qwen3 smoke passed locally on
  2026-08-12; ordinary CI remains offline. The RC-tag two-repo validation
  summary (Linux image suite 495 passed / 2 skipped, FinAgentBench 149,
  FinRun mutations 11/11) is recorded below and shipped with `v0.1.0-rc.3`.
  The post-tag `main` Linux-image regression on 2026-08-13 passed 508 tests
  with 3 skipped; it is not evidence from the immutable RC tag
- No large-scale multi-day soak in the validated pack

### Validated integration evidence (controlled)

| Gate | Result | Evidence |
|------|--------|----------|
| Queue/worker multi-process Docker | PASS (`20260804T095357Z`) — worker-kill reclaim without manual redelivery; tenant leakage `0`; orphan chunks/vectors `0/0` | [QUEUE_WORKER_INTEGRATION.md](QUEUE_WORKER_INTEGRATION.md) |
| Provider resilience dual-API Docker | PASS (`docker_20260804T100817Z`) — logical calls `20` → physical attempts `25` (1.25×); unexpected failures `0`; context leakage `0` | [PROVIDER_RESILIENCE.md](PROVIDER_RESILIENCE.md) |
| RC-tag two-repo full validation | PASS (2026-08-12) — LumenFin 495/2 skip; FinAgentBench 149; FinRun mutations 11/11 | this document + CI / `scripts/run_tests.py` + `scripts/run_cross_repo_ci.py` |
| Post-tag LumenFin regression | PASS (2026-08-13) — current `main` Linux-image suite 508/3 skip | local closure run; not part of immutable `v0.1.0-rc.3` evidence |

Integration harnesses use deterministic embeddings / demo market providers and a
deterministic provider stub for fault injection. Combined observed inflight
across API containers may reach the **sum** of per-process bulkhead limits.

### Infrastructure

- Local/dev can use Milvus Lite (single-process file). Validated multi-process
  integration uses **Milvus Server** via Docker Compose
- SQLite is allowed only for `APP_ENV=test` (and explicit `MAS_ALLOW_SQLITE_DEV`
  in `dev`). Production/integration require PostgreSQL
- Request runtime state is isolated; shared providers still need bounded
  concurrency and operator monitoring

### Multi-tenancy

- API keys map to an `AuthenticatedPrincipal` with a fixed `tenant_id`
  (see [MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md))
- Jobs, checkpoints, and RAG lookups are tenant-scoped on authorized paths
- This is API-key authorization isolation, **not** OAuth/OIDC/RBAC or
  physical infrastructure isolation
- PostgreSQL Row-Level Security is **not** enabled
- Schema upgrade for existing PostgreSQL databases:
  `migrations/postgresql/003_add_tenant_ownership.sql` (legacy rows bind to
  `tenant_id='default'`)

### External providers

- LLM, embedding, SEC, and Yahoo/market sources have quota, auth, latency,
  and availability risks
- Transient provider failure can produce `incomplete_data`; operators must
  distinguish this from structural data absence
- Provider resilience (deadline, Retry-After, single retry owner, degraded
  fallback) is validated with a deterministic stub plus bounded synthetic live
  smoke — not a live soak
- Provider HTTP retry ≠ Redis job retry ≠ critic replan (different layers)

### Evidence and data

- Live structured-source citations do not have PDF `#pN` page anchors
- SEC taxonomy and issuer filing differences limit structured fact coverage
- Growth claims require comparable multi-period fundamentals
- RC company coverage is finite and does not represent the entire market

### Product and compliance

- Reports are research artifacts, **not investment advice**
- Human review remains required for material decisions
- FinAgentBench measures FinRun replay / execution reliability; it does not
  prove future investment performance or act as a third-party market benchmark
- Project-owned source is MIT licensed
- PyMuPDF licensing (AGPL-3.0 and/or commercial terms) remains a blocker for
  publishing the application image as a purely MIT artifact
- Compose references AGPL MinIO and source-available Redis 7.4; operators and
  distributors must follow their separate terms

## Scale-up prerequisites (beyond this RC)

1. External IdP/OIDC, RBAC, key rotation, and audit policy
2. Optional PostgreSQL RLS and physical per-tenant isolation
3. Managed Milvus (or equivalent) with capacity planning
4. Cross-process rate limits / circuit policy if multi-region
5. Provider SLO monitoring, soak, and live smoke in CI
6. Pinned release tags and artifact retention
7. Resolve PyMuPDF/MinIO/Redis distribution obligations before publishing
   container artifacts
