-- Additive tenant ownership columns for analysis jobs and workflow checkpoints.
-- Positioning: pre-production / RC schema migration applied automatically by
-- scripts/run_integration_migrations.py (same path as 001/002). It may also be
-- applied directly with psql. Not a zero-downtime online migration framework.
--
-- Legacy rows: existing tables gain tenant_id with DEFAULT 'default'. PostgreSQL
-- fills prior rows with that default when the column is added. This matches the
-- product default for MAS_API_KEY_TENANT_ID / MAS_RAG_TENANT_ID ("default").
-- Operators who previously used a non-default logical tenant must re-bind data
-- or principals explicitly; this migration does not invent per-row ownership.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS / CREATE INDEX IF NOT EXISTS.

ALTER TABLE analysis_jobs
    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

ALTER TABLE workflow_checkpoints
    ADD COLUMN IF NOT EXISTS tenant_id TEXT NOT NULL DEFAULT 'default';

CREATE INDEX IF NOT EXISTS ix_analysis_jobs_tenant_id ON analysis_jobs (tenant_id);
CREATE INDEX IF NOT EXISTS ix_workflow_checkpoints_tenant_id ON workflow_checkpoints (tenant_id);
