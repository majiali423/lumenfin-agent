# Production Boundaries and Limitations

LumenFin is a **portfolio release candidate** validated under controlled
multi-process and deterministic fault-injection conditions. These results
are **not** a certification of unrestricted production readiness.

| Status | Meaning |
|--------|---------|
| Controlled environment acceptance | Phase 3.2B / 3.3A / offline regression PASS |
| Suitable for | Internal demos, interview portfolio, limited operator-owned deploys |
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
- Bulkheads are **per-process**, not a cross-process global rate limit
- No shared circuit breaker across API processes
- Live provider smoke is currently **skipped** in CI/release evidence
- No large-scale multi-day soak in the validated pack

### Infrastructure

- Local/dev can use Milvus Lite (single-process file). Validated multi-process
  integration uses **Milvus Server** via Docker Compose
- SQLite is allowed only for `APP_ENV=test` (and explicit `MAS_ALLOW_SQLITE_DEV`
  in `dev`). Production/integration require PostgreSQL
- Request runtime state is isolated; shared providers still need bounded
  concurrency and operator monitoring

### Multi-tenancy

- RAG data-plane is tenant-aware (logical isolation). See
  [MULTI_TENANCY_BOUNDARY.md](MULTI_TENANCY_BOUNDARY.md)
- Tenant identity is **not** bound to login credentials / JWT claims
- Checkpoint and analysis jobs are **not** fully tenant-scoped
- PostgreSQL Row-Level Security is **not** enabled

### External providers

- LLM, embedding, SEC, and Yahoo/market sources have quota, auth, latency,
  and availability risks
- Transient provider failure can produce `incomplete_data`; operators must
  distinguish this from structural data absence
- Provider resilience (deadline, Retry-After, single retry owner, degraded
  fallback) is validated with a deterministic stub — not a live soak

### Evidence and data

- Live structured-source citations do not have PDF `#pN` page anchors
- SEC taxonomy and issuer filing differences limit structured fact coverage
- Growth claims require comparable multi-period fundamentals
- RC company coverage is finite and does not represent the entire market

### Product and compliance

- Reports are research artifacts, **not investment advice**
- Human review remains required for material decisions
- FinAgentBench measures execution reliability; it does not prove future
  investment performance
- No public license grant is selected for this portfolio release
- PyMuPDF licensing (AGPL and/or commercial terms) remains a blocker for
  public Docker image redistribution until an explicit compliance path is chosen

## Scale-up prerequisites (beyond this RC)

1. Identity-bound tenant authorization (API keys / JWT claims)
2. Checkpoint/job tenant scoping + optional PostgreSQL RLS
3. Managed Milvus (or equivalent) with capacity planning
4. Cross-process rate limits / circuit policy if multi-region
5. Provider SLO monitoring, soak, and live smoke in CI
6. Pinned release tags and artifact retention
7. License decision for PDF stack redistribution
