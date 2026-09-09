-- Additive execution lease + outbox columns for analysis_jobs.
-- Positioning: pre-production / RC schema migration applied automatically by
-- scripts/run_integration_migrations.py (same path as 001/002/003). It may also
-- be applied directly with psql. Not a zero-downtime online migration framework.
--
-- Legacy rows: existing jobs gain delivery_state='published' so they are not
-- re-enqueued by the outbox republisher. New jobs start as delivery_state='pending'
-- until Redis publish succeeds. Execution token/owner/attempt/expires are NULL/0
-- until a worker claims the row.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS.
-- Rollback: leave the extra columns in place (do not DROP; they are additive and
-- unused columns do not change at-least-once delivery). Startup fails fast if an
-- existing non-SQLite analysis_jobs table is missing these columns.

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS execution_token TEXT;

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS execution_owner TEXT;

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS execution_attempt INTEGER NOT NULL DEFAULT 0;

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS execution_expires BIGINT;

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS delivery_state TEXT NOT NULL DEFAULT 'published';

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS delivery_payload_json TEXT;
