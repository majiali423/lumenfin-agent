-- Phase 3.2A: recover abandoned RAG indexing with owner fencing.

ALTER TABLE rag_documents
ADD COLUMN IF NOT EXISTS index_owner TEXT;

ALTER TABLE rag_documents
ADD COLUMN IF NOT EXISTS index_lease_expires BIGINT;

ALTER TABLE rag_documents
ADD COLUMN IF NOT EXISTS index_attempt INTEGER NOT NULL DEFAULT 0;
