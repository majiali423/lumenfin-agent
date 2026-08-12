# Milvus 3 cutover

Native BM25 is a separate schema migration after this Milvus 3 baseline. See
`docs/BM25_CUTOVER.md`; do not enable BM25 against `lumenfin_chunks_v3`.

The production Compose configuration uses Milvus Server 3.0 with PyMilvus 3.0.
Its versioned volumes and `lumenfin_chunks_v3` collection intentionally do not
reuse the former Milvus 2.4 storage.

## Why a rebuild is required

Milvus documents validate direct standalone upgrades from 2.6 to 3.0, not from
2.4. LumenFin therefore rebuilds vectors from the durable `rag_chunks` rows in
Postgres. Existing 2.4 volumes remain available for rollback until the cutover
has been accepted.

## Safe sequence

1. Stop API and index-worker writes. Keep Postgres available.
2. Back up Postgres and retain the old Milvus 2.4 volumes.
3. Start the new etcd, MinIO, and Milvus 3 services.
4. Run the read-only preflight:

   ```bash
   python scripts/rebuild_rag_vector_index.py
   ```

5. Confirm that the reported document and chunk counts match Postgres.
6. Reset and rebuild only the explicitly configured staging collection:

   ```bash
   python scripts/rebuild_rag_vector_index.py \
     --execute \
     --confirm-reset lumenfin_chunks_v3
   ```

7. Require `documents_indexed == documents_verified` and verify a real first
   RAG query before reopening writes.

The command refuses to reset an empty source corpus, refuses a mismatched
confirmation name, validates every ready document's durable chunk count before
reset, and does not allow a tenant-scoped rebuild to reset a shared collection.

## Rollback boundary

Do not point Milvus 2.4 at the new volumes and do not point Milvus 3 at the old
volumes. Before new writes are accepted, rollback is a configuration switch to
the untouched 2.4 stack. After new writes are accepted, restore from the
cutover backup or repeat the controlled rebuild; do not attempt an image-only
downgrade.
