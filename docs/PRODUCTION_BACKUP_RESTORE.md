# Production backup and restore

## Scope

The authoritative source set is PostgreSQL plus `uploads/`. Redis stores durable
queue state through AOF. The etcd, MinIO, and Milvus volumes provide a fast
rollback snapshot, but the vector/BM25 collection remains rebuildable from
PostgreSQL `rag_chunks`.

No backup or restore command deletes a Docker volume or a Milvus collection.

## Create and verify a backup

Run from the repository root while the production Compose project is healthy:

```powershell
powershell -File scripts/backup_production.ps1
powershell -File scripts/verify_production_backup.ps1 `
  -BackupDir backups/production-YYYYMMDD-HHMMSS
```

The backup script briefly stops API and worker writers, creates a custom-format
PostgreSQL dump, archives `uploads/` and `outputs/`, pauses stateful services,
and takes read-only snapshots of the Redis, etcd, MinIO, and Milvus named
volumes. It then restores the prior running state. Each artifact is recorded
with a SHA-256 digest in `manifest.json`.

Use `-SkipDerivedVolumes` only when PostgreSQL plus uploads are sufficient and a
Milvus rebuild is explicitly acceptable.

## Restore preconditions

Restore is intentionally operator-driven because it replaces state:

1. Obtain explicit approval for the target environment and backup timestamp.
2. Verify the backup with `verify_production_backup.ps1`.
3. Stop API and workers to freeze writes.
4. Create a second backup of the current state.
5. Restore into a new empty database/volume set first; do not overwrite the
   current volumes during a rehearsal.

## PostgreSQL rehearsal

Create an isolated PostgreSQL container or database, then run:

```powershell
pg_restore --clean --if-exists --no-owner --no-privileges `
  --dbname <isolated-target-url> backups/production-YYYYMMDD-HHMMSS/postgres.dump
```

Validate the `analysis_jobs`, `workflow_checkpoints`, `rag_documents`, and
`rag_chunks` tables before directing any application service to the restored
database.

## Milvus recovery

Prefer deterministic rebuild from restored PostgreSQL chunks:

```powershell
python scripts/rebuild_rag_vector_index.py --preflight-only
python scripts/rebuild_rag_vector_index.py `
  --execute --confirm-reset lumenfin_chunks_v4_bm25
python scripts/verify_rag_first_search.py
```

The destructive `--execute --confirm-reset` step requires separate approval.
Keep `lumenfin_chunks_v3` and the current v4 collection until the rebuilt
collection passes first-search and BM25 gates.

Raw volume archives are for fast same-version disaster recovery only. Restore
etcd, MinIO, and Milvus as one coordinated snapshot into newly created named
volumes; never mix timestamps or restore one of the three alone.

## Redis and uploads

Restore the Redis AOF archive only into a new Redis volume while Redis is
stopped. Replayed jobs retain at-least-once semantics and may be delivered
again, so verify idempotency and dead-letter depth after startup.

Restore `uploads.zip` before rebuilding documents whose original binary is
required. PostgreSQL chunks can reproduce search evidence, but cannot recreate
the original uploaded file byte-for-byte.

## Rollback

- Qwen3 to lexical: set `MAS_RAG_RERANK_PROVIDER=lexical`; no index rebuild.
- BM25 v4 to dense v3: follow `docs/BM25_CUTOVER.md`; retain both collections.
- Failed migration: the migrator blocks API/workers through
  `service_completed_successfully`.
- Failed restore rehearsal: discard only the newly created isolated target
  after explicit approval; leave current production state untouched.
