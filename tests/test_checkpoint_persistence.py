from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.checkpoint_store import WorkflowCheckpointRepository, infer_last_node
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.service import LumenFinAnalysisService
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class CheckpointPersistenceTestCase(unittest.TestCase):
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
