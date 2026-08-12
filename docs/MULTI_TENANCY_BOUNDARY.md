# Multi-Tenancy Boundary

**Scope:** authentication-bound tenant authorization for API callers, plus
RAG / job / checkpoint **logical isolation**. This is **not** full SaaS IAM,
OIDC, or physical infrastructure isolation.

---

## 1. Goal

Prevent tenant A from:

1. Impersonating tenant B by sending `tenant_id=tenant-b` on a request
2. Reading tenant B’s jobs, checkpoints, or indexed documents by ID

## 2. Identity chain

```
X-API-Key
  → AuthenticatedPrincipal(client_id, tenant_id)   # from server config
  → resolve_effective_tenant(request.tenant_id?)   # mismatch → HTTP 403
  → service / queue / checkpoint / RAG lookup
```

Configuration:

| Env | Role |
|-----|------|
| `MAS_API_KEY` | Legacy single key (bound to one tenant) |
| `MAS_API_KEY_CLIENT_ID` | Client id for the legacy key |
| `MAS_API_KEY_TENANT_ID` | Tenant bound to the legacy key (defaults toward `MAS_RAG_TENANT_ID`) |
| `MAS_API_KEY_PRINCIPALS` | JSON map of key → `{client_id, tenant_id}` |

Legacy keys cannot freely impersonate other tenants. Explicit request
`tenant_id` must match the principal or the API returns **403**.

## 3. Resource ownership

| Resource | Lookup | Cross-tenant result |
|----------|--------|---------------------|
| Analysis job | `job_id` + authorized `tenant_id` | **404** |
| Job list | filtered by `tenant_id` | foreign jobs omitted |
| Checkpoint / clarify | `thread_id` + authorized `tenant_id` | **404** |
| RAG document status | `document_id` + authorized `tenant_id` | **404** |
| Document index form `tenant_id` | must match principal | **403** on mismatch |

Queue / worker payloads carry the already-authorized `tenant_id` from the
API boundary; workers trust that internal payload, not a raw caller field.

## 4. RAG data plane (unchanged shape)

- Config default: `MAS_RAG_TENANT_ID`
- Canonical document IDs include `tenant_id + content_hash`
- Repository CRUD and Milvus metadata filters remain tenant-scoped
- Index jobs include `tenant_id` in Redis payloads

## 5. What this is / is not

| Layer | Status |
|-------|--------|
| Logical isolation (filters / metadata) | yes |
| Authorization isolation (principal-bound tenant) | yes |
| Physical isolation (separate DB/cluster per tenant) | no |
| OAuth / OIDC / Keycloak | no |
| Full RBAC / roles | no |
| PostgreSQL Row-Level Security | no |

## 6. Evidence

- Unit / API tests: `tests/test_tenant_authz.py`
- RAG tenant isolation harnesses remain under the offline suite
- Multi-process queue isolation evidence lives under integration validation
  docs (see reliability / validation command references)

## 7. Evolution path (out of scope here)

1. External IdP (OIDC) issuing tenant claims
2. Optional PostgreSQL RLS as defense in depth
3. Optional per-tenant collections / databases for stronger blast-radius limits

---

## Threat boundary table

| Risk | Protection | Status |
|------|------------|--------|
| A reads B’s RAG documents | tenant-scoped repository + authz tenant | covered |
| A retrieves B’s vectors | Milvus tenant filter + authorized tenant | covered |
| Redis worker loses tenant context | `tenant_id` in payload | covered |
| User forges `tenant_id` | principal binding → **403** | covered |
| Cross-tenant job / checkpoint | tenant-scoped lookup → **404** | covered |
| DB query forgets tenant filter | repository encapsulation + tests; no RLS | partially covered |
| Compromised API key for tenant A | full access within A only | by design |
