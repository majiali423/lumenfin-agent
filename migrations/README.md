# Database migrations

LumenFin creates new database tables from its SQLAlchemy models. Existing PostgreSQL
databases require explicit migrations when a model adds a column; the application does
not run PostgreSQL `ALTER TABLE` statements at startup.

Apply the checkpoint revision migration before starting an upgraded deployment:

```powershell
psql $env:MAS_DATABASE_URL -v ON_ERROR_STOP=1 -f migrations/postgresql/001_add_workflow_checkpoint_revision.sql
```

The migration is safe to run repeatedly. Startup fails with an actionable error if an
existing non-SQLite `workflow_checkpoints` table does not contain `revision`.
