# Multi-Tenancy Boundary

**Scope:** RAG data-plane **tenant-aware logical isolation** for document
indexing and retrieval. This is **not** a full SaaS IAM implementation.

Evidence (Phase 3.2B integration): tenant leakage count = **0**
(`outputs/phase32b_integration/20260804T095357Z/summary.json`).

---

## 1. Current multi-tenant goal

Prevent tenant A from reading tenant B’s indexed documents, chunks, or vectors
when callers supply distinct `tenant_id` values through repository and retrieval
APIs.

## 2. `tenant_id` source

- Config default: `MAS_RAG_TENANT_ID` (often `default`)
- Upload / index API: optional `tenant_id` form field overrides for that job
- Redis index payload: carries `tenant_id` for worker processing
- Callers are trusted to pass the intended tenant — **no login/JWT binding**

## 3. Document ID namespace

Canonical document IDs include **`tenant_id + content_hash`** so identical bytes
under different tenants do not collide.

## 4. PostgreSQL isolation

RAG document / chunk CRUD filters by `tenant_id` in the repository layer.
Not enforced via PostgreSQL Row-Level Security.

## 5. Redis payload

Index jobs include `tenant_id` in the message payload so workers restore
tenant context after reserve/reclaim.

## 6. Worker tenant context

Index workers read `tenant_id` from the job payload and pass it into claim /
index / finalize paths. Lost context without payload would be a defect; covered
by integration scenarios.

## 7. Milvus metadata filtering

Vector rows store tenant metadata; search expressions push down `tenant_id`
filters. Row keys are tenant-aware.

## 8–9. Keyword / hybrid retrieval filtering

Keyword and hybrid retrieval are tenant-scoped (repository + vector filter).

## 10. Integration evidence

| Check | Result |
|-------|--------|
| Phase 3.2B tenant isolation | PASS |
| `tenant_leakage_count` | 0 |
| Run id | `20260804T095357Z` |

## 11. Authentication gaps (not covered)

- No binding of `tenant_id` to authenticated principal
- No per-tenant API keys or JWT claims
- A caller who can hit the API can attempt to pass another tenant’s id

## 12. Checkpoint / analysis job gaps (not covered)

- Workflow checkpoints are not fully tenant-scoped
- Analysis job tables are not a complete multi-tenant IAM boundary

## 13. Production evolution path

1. Authenticate callers; derive `tenant_id` from claims (ignore client spoof)
2. Scope checkpoints and analysis jobs by tenant
3. Optional PostgreSQL RLS as defense in depth
4. Optional per-tenant collections / databases for stronger blast-radius limits

---

## Threat boundary table

| Risk | Current protection | Status |
|------|--------------------|--------|
| A reads B’s RAG documents | tenant-scoped repository query | covered |
| A retrieves B’s vectors | Milvus tenant filter | covered |
| Redis worker loses tenant context | `tenant_id` in payload | covered |
| User forges `tenant_id` | no identity binding | **not covered** |
| Cross-tenant checkpoint | checkpoint not fully tenant scoped | **not covered** |
| DB query forgets tenant filter | repository encapsulation + tests; no RLS | **partially covered** |
