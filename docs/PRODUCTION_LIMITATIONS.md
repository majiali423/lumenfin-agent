# Production Boundaries and Limitations

> Ready for controlled production deployment, subject to documented
> infrastructure and provider constraints.

This statement does not mean unlimited or internet-scale production readiness.

## Current suitable scope

- Controlled internal deployment
- Single instance or limited workers with explicit process ownership
- Services with provider quotas, monitoring and operator response
- Financial research support with human review
- Replayable, audited Agent runs

## Current non-goals

- Internet-scale multi-tenant high availability
- Strongly consistent distributed checkpoints
- Large managed Milvus clusters
- Zero-downtime model-provider switching
- High-frequency trading
- Automated investment, legal or fiduciary decisions
- Trade execution

## Known limitations

### Infrastructure

- Milvus Lite is a single-machine facility. It is not shared multi-writer
  production vector infrastructure.
- SQLite is not an HA multi-tenant checkpoint/job store.
- Request runtime state is isolated, but shared provider/RAG infrastructure
  still needs bounded concurrency and load testing.

### External providers

- LLM, embedding, SEC and Yahoo/market sources have quota, authentication,
  latency and availability risks.
- Model names and availability can change.
- Transient provider failure can produce `incomplete_data`; operations must
  distinguish this from structural data absence.

### Evidence and data

- Live structured-source citations do not have PDF `#pN` page anchors.
- SEC taxonomy and issuer filing differences limit structured fact coverage.
- Growth claims require comparable multi-period fundamentals and are rejected
  when those inputs are absent.
- RC company coverage is finite and does not represent the entire market.

### Product and compliance

- Reports are research artifacts, not investment advice.
- Human review remains required for material decisions.
- FinAgentBench measures execution reliability; it does not prove future
  investment performance or universal factual correctness.
- No public license grant is selected for this internal/portfolio release.
- PyMuPDF licensing (AGPL and/or commercial terms) remains a blocker for
  public Docker image redistribution and any public binary redistribution
  that embeds the library until an explicit compliance path is chosen.

## Scale-up prerequisites

Before broader production use:

1. Managed Milvus or equivalent vector service
2. PostgreSQL/native distributed checkpoint strategy
3. Queue backpressure and worker ownership
4. Provider SLO/health monitoring and quota alarms
5. Load, soak and failure-injection testing
6. Pinned release tags and artifact retention
7. Tenant authorization and data-retention controls
