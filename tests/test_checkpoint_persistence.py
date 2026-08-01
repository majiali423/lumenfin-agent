from __future__ import annotations

import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier
from unittest.mock import Mock, patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.checkpoint_store import WorkflowCheckpointRepository, infer_last_node
from lumenfin.database import Base
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.service import LumenFinAnalysisService
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class CheckpointPersistenceTestCase(unittest.TestCase):
    def test_postgresql_missing_revision_fails_fast_with_migration_path(self) -> None:
        engine = Mock()
        engine.url = "postgresql+psycopg://db.example/lumenfin"
        schema = Mock()
        schema.has_table.return_value = True
        schema.get_columns.return_value = [{"name": "thread_id"}, {"name": "updated_at"}]

        with patch.object(Base.metadata, "create_all"), patch(
            "lumenfin.checkpoint_store.inspect",
            return_value=schema,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"workflow_checkpoints\.revision.*001_add_workflow_checkpoint_revision\.sql.*psql",
            ):
                WorkflowCheckpointRepository(engine)

    def test_postgresql_revision_migration_is_repeatable(self) -> None:
        migration = ROOT / "migrations" / "postgresql" / "001_add_workflow_checkpoint_revision.sql"
        sql = migration.read_text(encoding="utf-8")
        self.assertIn("ADD COLUMN IF NOT EXISTS revision", sql)
        self.assertIn("INTEGER NOT NULL DEFAULT 1", sql)

    def test_same_base_revision_allows_one_writer_and_latest_can_retry(self) -> None:
        root = ROOT / "test_artifacts" / f"checkpoint-cas-{uuid4().hex[:8]}"
        config = build_test_config(root)
        repo = WorkflowCheckpointRepository.from_database_url(config.database_url, db_path=config.db_path)
        initial = repo.upsert(
            thread_id="cas-thread",
            query="initial",
            state={"workflow_status": "needs_clarification", "query": "initial"},
            expected_revision=0,
        )
        base_revision = initial["revision"]
        barrier = Barrier(2)

        def write(query: str) -> tuple[str, object]:
            barrier.wait(timeout=10)
            try:
                return "ok", repo.upsert(
                    thread_id="cas-thread",
                    query=query,
                    state={"workflow_status": "completed", "query": query},
                    expected_revision=base_revision,
                )
            except Exception as exc:  # noqa: BLE001 - asserting public conflict type
                return "error", exc

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(write, ["first", "second"]))

        successes = [value for status, value in outcomes if status == "ok"]
        errors = [value for status, value in outcomes if status == "error"]
        self.assertEqual(len(successes), 1)
        self.assertEqual(len(errors), 1)
        self.assertEqual(type(errors[0]).__name__, "CheckpointConflictError")

        stored = repo.get("cas-thread")
        self.assertEqual(stored["query"], successes[0]["query"])
        self.assertEqual(stored["state"]["query"], successes[0]["state"]["query"])
        retried = repo.upsert(
            thread_id="cas-thread",
            query="retry",
            state={"workflow_status": "completed", "query": "retry"},
            expected_revision=stored["revision"],
        )
        self.assertEqual(retried["query"], "retry")
        self.assertEqual(retried["revision"], stored["revision"] + 1)

    def test_infer_last_node_for_hitl_pause(self) -> None:
        node = infer_last_node({"workflow_status": "needs_clarification", "audit_log": [{"step": "await_clarification"}]})
        self.assertEqual(node, "await_clarification")

    def test_hitl_survives_service_restart(self) -> None:
        root = ROOT / "test_artifacts" / f"checkpoint-{uuid4().hex[:8]}"
        config = build_test_config(root)
        repo = WorkflowCheckpointRepository.from_database_url(config.database_url, db_path=config.db_path)

        service1 = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
            checkpoint_repo=repo,
        )
        paused = service1.analyze(
            "请分析供应链风险和研发投入。",
            thread_id="hitl-persist",
            export_artifacts=False,
        )
        self.assertEqual(paused["workflow_status"], "needs_clarification")
        stored = repo.get("hitl-persist")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["workflow_status"], "needs_clarification")
        self.assertEqual(stored["last_node"], "await_clarification")

        service2 = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
            checkpoint_repo=repo,
        )
        resumed = service2.clarify(
            "hitl-persist",
            {"company": "Apple", "time_range": "FY2025"},
            export_artifacts=False,
        )
        self.assertEqual(resumed["workflow_status"], "completed")
        self.assertIn("Apple", resumed["result"]["final_report"])


if __name__ == "__main__":
    unittest.main()
