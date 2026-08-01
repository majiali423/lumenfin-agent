-- Phase 3.1: optimistic concurrency revision for persisted workflow checkpoints.
-- Repeatable: safe for databases where the column has already been added.
ALTER TABLE workflow_checkpoints
ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 1;
