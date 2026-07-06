from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.config import AppConfig
from lumenfin.graph import (
    route_after_critic,
    route_after_repair,
)
from lumenfin.critic_repair import classify_critic_repair_target
from lumenfin.llm import LLMSettings


def build_test_config(root: Path) -> AppConfig:
    db_path = root / "data" / "lumenfin.db"
    return AppConfig(
        output_dir=root / "outputs",
        upload_dir=root / "uploads",
        db_path=db_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        redis_url=None,
        redis_queue_name="finance-analysis-test",
        neo4j_uri=None,
        neo4j_username=None,
        neo4j_password=None,
        market_data_provider="fake",
        alphavantage_api_key=None,
        host="127.0.0.1",
        port=8000,
        api_key=None,
        llm=LLMSettings(api_key=None, base_url="https://api.deepseek.com", model="deepseek-chat", timeout_seconds=45),
        rag_enabled=True,
        milvus_uri=str(root / "data" / f"milvus_{uuid4().hex[:8]}.db"),
        milvus_collection="lumenfin_chunks_test",
        embedding_provider="deterministic",
        embedding_dimension=384,
        rag_top_k=5,
        critic_max_iterations=2,
        company_parallelism=4,
        input_guardrail_enabled=True,
        input_guardrail_mode="sanitize",
        tool_backend="local",
    )


class GraphRoutingTestCase(unittest.TestCase):
    def test_critic_routes_to_synthesizer_when_clean(self) -> None:
        state = {"compliance_findings": [], "critic_iterations": 0, "critic_max_iterations": 2}
        self.assertEqual(route_after_critic(state), "synthesizer")

    def test_critic_routes_to_repair_when_findings_remain(self) -> None:
        state = {
            "compliance_findings": ["Apple: missing quantitative results."],
            "critic_iterations": 0,
            "critic_max_iterations": 2,
        }
        self.assertEqual(route_after_critic(state), "repair")

    def test_critic_routes_to_synthesizer_after_max_iterations(self) -> None:
        state = {
            "compliance_findings": ["Apple: missing quantitative results."],
            "critic_iterations": 2,
            "critic_max_iterations": 2,
        }
        self.assertEqual(route_after_critic(state), "synthesizer")

    def test_classify_repair_target_for_quant_gap(self) -> None:
        findings = ["Apple: missing quantitative results."]
        self.assertEqual(classify_critic_repair_target(findings), "quant")

    def test_classify_repair_target_for_sentiment_gap(self) -> None:
        findings = ["Apple: missing sentiment analysis."]
        self.assertEqual(classify_critic_repair_target(findings), "psychologist")

    def test_repair_routes_to_configured_target(self) -> None:
        state = {"critic_repair_target": "quant"}
        self.assertEqual(route_after_repair(state), "quant")


if __name__ == "__main__":
    unittest.main()
