# RAG indexing lease

Upload-time document indexing uses a database lease to recover work abandoned while
`rag_documents.index_status` is `indexing`.

`MAS_RAG_INDEX_LEASE_SECONDS` controls the lease duration and defaults to 300 seconds.
Each claim stores a random `index_owner`, an expiry epoch in `index_lease_expires`, and
increments the monotonic `index_attempt` fencing generation. Pending work and indexing
work whose lease has expired can be claimed through one atomic database update.

The indexer renews the lease at stage boundaries: before parsing, before vector writes,
before chunk persistence, and before the final ready transition. Ready and failed
transitions require the current owner and attempt. Failure cleanup is skipped when that
ownership check is lost, preventing a stale worker from deleting a newer worker's data.

This is stage-boundary lease renewal. It does not claim uninterrupted heartbeats,
exactly-once external side effects, or protection for an indefinitely long individual
provider call. Choose a lease duration longer than expected parse/embed/vector calls.

Existing SQLite databases receive the three columns through the repository's additive
SQLite migration. Existing PostgreSQL databases must apply
`migrations/postgresql/002_add_rag_index_lease.sql` before startup.
