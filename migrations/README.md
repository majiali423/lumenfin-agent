# Database migrations

LumenFin creates new database tables from its SQLAlchemy models. Existing PostgreSQL
databases require explicit migrations when a model adds a column; the application does
not run PostgreSQL `ALTER TABLE` statements at startup.

Apply the checkpoint revision migration before starting an upgraded deployment:

```powershell
$psqlUrl = $env:MAS_DATABASE_URL `
    -replace '^postgresql\+psycopg://', 'postgresql://'

psql $psqlUrl `
    -v ON_ERROR_STOP=1 `
    -f migrations/postgresql/001_add_workflow_checkpoint_revision.sql

psql $psqlUrl `
    -v ON_ERROR_STOP=1 `
    -f migrations/postgresql/002_add_rag_index_lease.sql
```

`MAS_DATABASE_URL` uses SQLAlchemy's `postgresql+psycopg://` scheme, while `psql`
accepts the libpq `postgresql://` (or `postgres://`) scheme. The replacement changes
only the driver scheme; credentials, host, port, and database name are preserved.

The migration is safe to run repeatedly. Startup fails with an actionable error if an
existing non-SQLite `workflow_checkpoints` table does not contain `revision`, or if
`rag_documents` is missing the indexing lease columns.
