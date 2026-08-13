from __future__ import annotations

import importlib.util
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_migration_runner():
    path = ROOT / "scripts" / "run_integration_migrations.py"
    spec = importlib.util.spec_from_file_location("run_integration_migrations_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PostgreSQLMigrationManifestTests(unittest.TestCase):
    def test_automatic_manifest_contains_all_migrations_in_order(self) -> None:
        runner = _load_migration_runner()

        self.assertEqual(
            [path.name for path in runner.MIGRATIONS],
            [
                "001_add_workflow_checkpoint_revision.sql",
                "002_add_rag_index_lease.sql",
                "003_add_tenant_ownership.sql",
            ],
        )

    def test_tenant_migration_is_repeat_safe_for_both_owned_tables(self) -> None:
        sql = (
            ROOT / "migrations" / "postgresql" / "003_add_tenant_ownership.sql"
        ).read_text(encoding="utf-8")

        self.assertIn("ALTER TABLE analysis_jobs", sql)
        self.assertIn("ALTER TABLE workflow_checkpoints", sql)
        self.assertEqual(sql.count("ADD COLUMN IF NOT EXISTS tenant_id"), 2)
        self.assertIn("DEFAULT 'default'", sql)
        self.assertEqual(
            len(re.findall(r"^CREATE INDEX IF NOT EXISTS ", sql, flags=re.MULTILINE)),
            2,
        )


if __name__ == "__main__":
    unittest.main()
