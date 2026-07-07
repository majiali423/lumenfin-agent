from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from uuid import uuid4

import fitz
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.api.app import create_app
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.reporting import export_run_artifacts
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


def build_offline_system(config=None) -> LumenFinAgentSystem:
    app_config = config or build_test_config(ROOT / "test_artifacts" / f"offline-{uuid4().hex[:8]}")
    return LumenFinAgentSystem(
        llm_client=LocalFallbackLLMClient(),
        app_config=app_config,
        market_data_client=FakeMarketDataClient(),
    )


class OfflineSystemTestCase(unittest.TestCase):
    def test_end_to_end_report_generation(self) -> None:
        app = build_offline_system()
        result = app.run("对比分析 Apple 与 Microsoft 2025 年供应链风险和研发投入。", thread_id="test-e2e-offline")

        self.assertIn("final_report", result)
        self.assertIn("Apple", result["final_report"])
        self.assertIn("Microsoft", result["final_report"])
        self.assertEqual(result["llm_backend"], "local-fallback")

        steps = [event["step"] for event in result["audit_log"]]
        for required in ("input_guardrail", "query_planner", "supervisor", "retrieval", "quant", "psychologist", "critic", "synthesizer"):
            self.assertIn(required, steps)

    def test_replanner_path_and_exports(self) -> None:
        app = build_offline_system()
        result = app.run("分析 Apple 2025 财报的供应链附录风险。", thread_id="test-replanner-offline")

        steps = [event["step"] for event in result["audit_log"]]
        self.assertIn("retrieval", steps)
        self.assertIn("quant", steps)

        with tempfile.TemporaryDirectory() as tmp_dir:
            artifacts = export_run_artifacts(result, Path(tmp_dir), "test-replanner-offline")
            for artifact_path in artifacts.values():
                self.assertTrue(Path(artifact_path).exists())

            state_payload = json.loads(Path(artifacts["state_path"]).read_text(encoding="utf-8"))
            self.assertEqual(state_payload["thread_id"], "test-replanner-offline")

    def test_api_endpoint_offline(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"api-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = create_app(
            build_test_config(tmp_root),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze",
                json={
                    "query": "对比分析 Apple 与 Microsoft 2025 年供应链风险和研发投入。",
                    "thread_id": "test-api-offline",
                    "export_artifacts": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["thread_id"], "test-api-offline")
            self.assertIn("final_report", payload)
            self.assertTrue(payload["final_report"])

    def test_upload_endpoint_offline(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"upload-test-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = create_app(
            build_test_config(tmp_root),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with TestClient(app) as client:
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "Apple revenue 400 EBITDA 120 risk warning")
            pdf_bytes = pdf.tobytes()
            pdf.close()
            response = client.post(
                "/api/v1/analyze-upload",
                data={
                    "query": "请分析这份 Apple 财报 PDF 的核心风险。",
                    "thread_id": "upload-test-offline",
                    "export_artifacts": "false",
                },
                files={"files": ("Apple_report.pdf", pdf_bytes, "application/pdf")},
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("Apple", payload["final_report"])

    def test_upload_csv_endpoint_offline(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"upload-csv-{uuid4().hex[:8]}"
        tmp_root.mkdir(parents=True, exist_ok=True)
        app = create_app(
            build_test_config(tmp_root),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        csv_body = (
            "company,revenue_2025,ebitda_2025\n"
            "NVIDIA,130.5,75.2\n"
        ).encode("utf-8")
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/analyze-upload",
                data={
                    "query": "请基于上传的 NVIDIA 结构化指标输出尽调速写。",
                    "thread_id": "upload-csv-offline",
                    "export_artifacts": "false",
                },
                files={"files": ("nvidia_metrics.csv", csv_body, "text/csv")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            state = payload.get("state") or {}
            metrics = state.get("financial_metrics") or {}
            self.assertIn("NVIDIA", metrics)
            self.assertAlmostEqual(metrics["NVIDIA"]["ebitda_margin"], round(75.2 / 130.5, 4))
            self.assertIn("NVIDIA", payload.get("final_report", ""))
            manifest = payload.get("run_manifest") or {}
            upload_formats = (manifest.get("data_sources") or {}).get("upload_formats") or []
            self.assertIn("csv", upload_formats)


@unittest.skipUnless(os.getenv("RUN_INTEGRATION_TESTS") == "1", "set RUN_INTEGRATION_TESTS=1 for live API tests")
class IntegrationSystemTestCase(unittest.TestCase):
    def test_end_to_end_with_live_fallback_chain(self) -> None:
        app = LumenFinAgentSystem()
        result = app.run("对比分析 Apple 与 Microsoft 的供应链风险和研发投入。", thread_id="test-integration")
        self.assertIn("final_report", result)
        self.assertIn(result["llm_backend"], {"deepseek", "local-fallback"})


if __name__ == "__main__":
    unittest.main()
