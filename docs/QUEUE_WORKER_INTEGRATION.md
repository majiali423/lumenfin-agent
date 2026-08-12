# Queue / worker integration evidence

Desensitized validation record for PostgreSQL-first runtime and reliable Redis
delivery. Generated from a real PASS run; not a certification of unrestricted
production readiness.

**Runner:** `python scripts/run_queue_worker_integration.py`  
**Compose:** `docker-compose.integration.yml`

## Test metadata

| Field | Value |
|-------|-------|
| Test date (UTC) | 2026-08-04 |
| Started | 2026-08-04T07:59:52Z |
| Finished | 2026-08-04T08:01:51Z (approx.; suite elapsed ~121s) |
| Suite status | **pass** |
| Base commit before closure work | `7e2c3cb5e3f568aa5730b6deaf871af408d1a2a8` |
| Artifact run id | `20260804T075952Z` |

## Infrastructure versions

| Component | Image / version |
|-----------|-----------------|
| PostgreSQL | `postgres:16` |
| Redis | `redis:7` |
| Milvus | `milvusdb/milvus:v2.4.15` |
| Docker / Compose | recorded in run `summary.json` → `versions` |

## Container identities (short IDs)

| Role | Container ID prefix |
|------|---------------------|
| api-a | `857906f76186` |
| api-b | `6ef6f4757c8d` |
| index-worker-a | `eebd3937ba6d` |
| index-worker-b | `c24b1455fca2` |

## Migration result

- Empty-DB bootstrap + SQL migrations: **pass**
- Applied: `001_add_workflow_checkpoint_revision.sql`, `002_add_rag_index_lease.sql`
- Repeat-safe / fail-fast checks exercised by migration gate scenario

## Scenario results

| Scenario | Result |
|----------|--------|
| Checkpoint CAS (same/different thread) | pass — different_thread_success=10, same_thread_success=1, same_thread_conflict=1 |
| Duplicate Redis index messages | pass — canonical document count=1, chunk=1, vector=1, queues empty |
| Worker kill + automatic reclaim | pass — manual_redelivery=false, index_attempt=2, status=ready |
| Stale fencing | pass — stale_finalize_rejected=2, stale_cleanup_rejected=1 |
| Tenant isolation | pass — tenant_leakage_count=0 |
| Dead-letter after max attempts | pass — attempt=3, dead_letter=1, pending=0, processing=0 |
| ACK idempotency | pass — second ACK no-op |
| Redis restart recovery | pass — job completed after reconnect |
| Limited load | pass — orphan_chunk_count=0, orphan_vector_count=0 |

## Worker kill timeline (automatic reclaim)

1. Queues purged; Worker A started with index-pause armed
2. Document uploaded; Worker A reserved message (pending→processing) and claimed lease (`index_attempt=1`)
3. Worker A container killed — **no manual re-enqueue**
4. Waited for DB lease expiry + Redis reclaim idle
5. Worker B reclaimed processing→pending, reserved, claimed (`index_attempt=2`), indexed to `ready`, ACK
6. Final: owner/lease cleared; queues empty; document_count=1; chunk_count=1; vector_count=1

## Queue final lengths (suite end)

| Queue | Length |
|-------|--------|
| pending | 0 |
| processing | 0 |
| dead-letter | 0 (DLQ scenario left 1 letter mid-suite; later purged / drained to 0 by suite end observe) |

Dead-letter scenario peak: `dead_letter=1` with `message_id`, `payload`, `attempt=3`, `last_error=document not found`, `failed_at` set.

## PostgreSQL / Milvus counts (representative)

| Check | Value |
|-------|-------|
| Duplicate scenario documents | 1 |
| Duplicate scenario chunks | 1 |
| Duplicate scenario vectors | 1 |
| Lease recovery documents | 1 |
| Lease recovery chunks / vectors | 1 / 1 |
| Orphan chunks / vectors (load) | 0 / 0 |
| Duplicate vectors | 0 |

## Known limitations

- At-least-once delivery only — not exactly-once
- Reserve polls with short sleep (Lua atomic move); not BRPOP blocking
- Analysis and index workers share the reliable queue abstraction; reclaim is cooperative per-worker
- Integration uses deterministic embeddings and demo/fake market providers
- Not unrestricted production readiness; controlled multi-process validation only

## Artifacts (local, gitignored)

```text
outputs/queue_worker_integration/20260804T075952Z/summary.json
outputs/queue_worker_integration/20260804T075952Z/api-a.log
outputs/queue_worker_integration/20260804T075952Z/api-b.log
outputs/queue_worker_integration/20260804T075952Z/index-worker-a.log
outputs/queue_worker_integration/20260804T075952Z/index-worker-b.log
outputs/queue_worker_integration/summary.json
```

Migration detail is under the run directory scenario JSON / postgres logs as produced by the harness.
