from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from lumenfin.api.app import create_app
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.monitoring import (
    instrument_workflow,
    observe_packaged_run,
    prometheus_http_middleware,
    refresh_queue_depths,
    render_metrics,
    start_worker_metrics_server,
)
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class PrometheusMonitoringTestCase(unittest.TestCase):
    def test_application_metrics_endpoint_is_scrapeable_without_api_key(self) -> None:
        app = create_app(
            build_test_config(ROOT / "test_artifacts" / "prometheus-endpoint"),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            response = client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertIn("lumenfin_http_requests_total", response.text)

    def test_http_middleware_uses_route_template_not_user_path(self) -> None:
        app = FastAPI()
        app.middleware("http")(prometheus_http_middleware)

        @app.get("/items/{item_id}")
        def item(item_id: str) -> dict[str, str]:
            return {"item_id": item_id}

        @app.get("/metrics")
        def metrics():
            return render_metrics()

        client = TestClient(app)
        secret = "tenant-sensitive-123"
        self.assertEqual(client.get(f"/items/{secret}").status_code, 200)
        payload = client.get("/metrics").text
        self.assertIn('route="/items/{item_id}"', payload)
        self.assertNotIn(secret, payload)

    def test_existing_trace_is_projected_without_sensitive_labels(self) -> None:
        secret = "private-query-and-tenant"
        observe_packaged_run(
            {
                "workflow_status": "completed",
                "llm_backend": "deepseek",
                "provider_degraded": {"provider": "deepseek"},
                "provider_call_summary": {
                    "logical_provider_calls": 2,
                    "successes": 2,
                    "retries": 1,
                    "fallbacks": 0,
                    "total_provider_latency_ms": 250,
                },
                "result": {
                    "query": secret,
                    "tenant_id": secret,
                    "critic_repair_target": "retrieval",
                    "structured_answer": {
                        "citation_validation": "passed",
                        "citations": ["doc:p1:c1"],
                    },
                    "run_telemetry": {
                        "node_spans": [
                            {"step": "retrieval", "status": "ok", "latency_ms": 40},
                            {"step": "synthesizer", "status": "ok", "latency_ms": 60},
                        ],
                        "rag": {
                            "mode": "hybrid_dense_bm25_rrf+qwen3_rerank",
                            "degraded": False,
                            "rerank_providers": ["qwen3"],
                            "rerank_latency_ms": 12,
                            "rerank_fallbacks": 0,
                        },
                    },
                },
            },
            operation="analyze",
            duration_seconds=0.2,
        )
        output = generate_latest().decode("utf-8")
        self.assertIn("lumenfin_workflow_runs_total", output)
        self.assertIn("lumenfin_node_duration_seconds_bucket", output)
        self.assertIn("lumenfin_provider_calls_total", output)
        self.assertIn("lumenfin_rag_queries_total", output)
        self.assertIn('status="valid"', output)
        self.assertNotIn(secret, output)
        self.assertNotIn("doc:p1:c1", output)

    def test_workflow_exception_is_recorded_and_reraised(self) -> None:
        @instrument_workflow("unit-test-operation")
        def explode() -> dict:
            raise ValueError("safe test failure")

        with self.assertRaises(ValueError):
            explode()
        output = generate_latest().decode("utf-8")
        self.assertIn('operation="unit-test-operation",status="error"', output)
        self.assertIn('error_type="ValueError",operation="unit-test-operation"', output)

    def test_queue_depth_refresh_uses_only_queue_and_state_labels(self) -> None:
        pipeline = MagicMock()
        pipeline.llen.return_value = pipeline
        pipeline.execute.return_value = [3, 2, 1, 0]
        client = MagicMock()
        client.pipeline.return_value = pipeline
        with patch("lumenfin.monitoring.Redis.from_url", return_value=client):
            refresh_queue_depths("redis://example.invalid/0", ["finance-analysis"])
        output = generate_latest().decode("utf-8")
        self.assertIn('queue="finance-analysis",state="pending"} 3.0', output)
        self.assertIn('queue="finance-analysis",state="dead-letter"} 1.0', output)
        client.close.assert_called_once()

    def test_worker_metrics_server_is_opt_in_by_port(self) -> None:
        with patch.dict(os.environ, {"MAS_METRICS_PORT": "0"}, clear=False):
            self.assertFalse(start_worker_metrics_server("test-worker"))


if __name__ == "__main__":
    unittest.main()
