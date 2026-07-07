from __future__ import annotations

import sys
import unittest
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.logging_utils import redact_secrets
from lumenfin.llm import LocalFallbackLLMClient
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class ObservabilityTestCase(unittest.TestCase):
    def test_redacts_secrets_from_logged_urls(self) -> None:
        raw = "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey=secret123"
        redacted = redact_secrets(raw)
        self.assertIn("apikey=[REDACTED]", redacted)
        self.assertNotIn("secret123", redacted)

    def test_redacts_secrets_from_url_like_objects(self) -> None:
        class UrlLike:
            def __str__(self) -> str:
                return "https://example.test/query?token=secret-token"

        redacted = redact_secrets(UrlLike())
        self.assertEqual(redacted, "https://example.test/query?token=[REDACTED]")

    def test_audit_log_contains_span_metrics(self) -> None:
        config = build_test_config(ROOT / "test_artifacts" / f"obs-{uuid4().hex[:8]}")
        app = LumenFinAgentSystem(
            llm_client=LocalFallbackLLMClient(),
            app_config=config,
            market_data_client=FakeMarketDataClient(),
        )
        result = app.run(
            "对比分析 Apple 与 Microsoft 2025 年供应链风险和研发投入。",
            thread_id="obs-test",
        )
        planner_events = [e for e in result.get("audit_log", []) if e.get("step") == "query_planner"]
        self.assertTrue(planner_events)
        event = planner_events[0]
        self.assertIn("latency_ms", event)
        self.assertIn("prompt_tokens", event)
        self.assertIn("completion_tokens", event)
        self.assertIn("estimated_cost_usd", event)

        telemetry = result.get("run_telemetry", {})
        self.assertIn("node_spans", telemetry)
        self.assertGreaterEqual(int(telemetry.get("total_prompt_tokens", 0)), 0)
        self.assertGreater(float(telemetry.get("total_latency_ms", 0.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
