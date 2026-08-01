from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch
from uuid import uuid4

import fitz
import httpx
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.api.app import create_app
from lumenfin.checkpoint_store import CheckpointConflictError
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.reporting import export_run_artifacts
from lumenfin.service import LumenFinAnalysisService
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
    def test_upload_file_persistence_does_not_block_lightweight_request(self) -> None:
        cases = [
            (
                "/api/v1/analyze-upload",
                {"query": "Analyze Apple FY2025.", "thread_id": "save-upload", "export_artifacts": "false"},
            ),
            ("/api/v1/documents/index", {"tenant_id": "save-tenant"}),
        ]

        for route, form_data in cases:
            with self.subTest(route=route):
                tmp_root = ROOT / "test_artifacts" / f"api-save-concurrency-{uuid4().hex[:8]}"
                app = create_app(
                    build_test_config(tmp_root),
                    llm_client=LocalFallbackLLMClient(),
                    market_data_client=FakeMarketDataClient(),
                )
                save_entered = Event()
                release_save = Event()
                lightweight_done = Event()
                observed = {"lightweight_before_release": False}
                loop_holder: dict[str, asyncio.AbstractEventLoop] = {}
                gate_holder: dict[str, asyncio.Event] = {}

                def blocked_save(_service, files):
                    self.assertTrue(files)
                    save_entered.set()
                    self.assertTrue(release_save.wait(timeout=10))
                    return [str(tmp_root / "saved.txt")]

                def fake_analyze(_service, *args, **kwargs):
                    thread_id = kwargs.get("thread_id") or "save-upload"
                    return {
                        "thread_id": thread_id,
                        "query": kwargs.get("query", ""),
                        "llm_backend": "local-fallback",
                        "workflow_status": "completed",
                        "checkpoint": None,
                        "provider_health": {},
                        "result": {
                            "thread_id": thread_id,
                            "workflow_status": "completed",
                            "final_report": "done",
                            "audit_log": [],
                            "run_telemetry": {},
                            "llm_backend": "local-fallback",
                        },
                        "artifacts": {},
                    }

                def fake_index(_service, paths, **kwargs):
                    return [
                        {
                            "document_id": "doc-save",
                            "tenant_id": kwargs.get("tenant_id") or "save-tenant",
                            "filename": "saved.txt",
                            "content_hash": "hash-save",
                            "status": "ready",
                            "chunk_count": 1,
                            "error": None,
                            "contexts": [],
                            "embed_calls": 1,
                        }
                    ]

                def controller() -> None:
                    self.assertTrue(save_entered.wait(timeout=10))
                    loop_holder["loop"].call_soon_threadsafe(gate_holder["gate"].set)
                    observed["lightweight_before_release"] = lightweight_done.wait(timeout=2)
                    release_save.set()

                async def scenario() -> tuple[httpx.Response, httpx.Response]:
                    transport = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                        loop_holder["loop"] = asyncio.get_running_loop()
                        gate_holder["gate"] = asyncio.Event()

                        async def lightweight_request() -> httpx.Response:
                            await gate_holder["gate"].wait()
                            response = await client.get("/api/v1/config")
                            lightweight_done.set()
                            return response

                        lightweight_task = asyncio.create_task(lightweight_request())
                        upload_task = asyncio.create_task(
                            client.post(
                                route,
                                data=form_data,
                                files={"files": ("notes.txt", b"Apple revenue FY2025 was 100 billion.", "text/plain")},
                            )
                        )
                        return await upload_task, await lightweight_task

                controller_thread = Thread(target=controller, daemon=True)
                controller_thread.start()
                with (
                    patch.object(LumenFinAnalysisService, "save_uploaded_files", blocked_save),
                    patch.object(LumenFinAnalysisService, "analyze", fake_analyze),
                    patch.object(LumenFinAnalysisService, "index_document_paths", fake_index),
                ):
                    upload_response, lightweight_response = asyncio.run(scenario())
                controller_thread.join(timeout=10)

                self.assertTrue(observed["lightweight_before_release"])
                self.assertEqual(lightweight_response.status_code, 200)
                self.assertEqual(upload_response.status_code, 200, upload_response.text)

    def test_upload_analysis_does_not_block_lightweight_request(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"api-upload-concurrency-{uuid4().hex[:8]}"
        app = create_app(
            build_test_config(tmp_root),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        analyze_entered = Event()
        release_analyze = Event()
        lightweight_done = Event()
        observed = {"lightweight_before_release": False}
        loop_holder: dict[str, asyncio.AbstractEventLoop] = {}
        lightweight_gate_holder: dict[str, asyncio.Event] = {}

        def blocked_analyze(_service, *args, **kwargs):
            analyze_entered.set()
            self.assertTrue(release_analyze.wait(timeout=10))
            thread_id = kwargs.get("thread_id") or "upload-concurrency"
            result = {
                "thread_id": thread_id,
                "workflow_status": "completed",
                "final_report": "Upload analysis completed.",
                "audit_log": [],
                "run_telemetry": {},
                "llm_backend": "local-fallback",
            }
            return {
                "thread_id": thread_id,
                "query": kwargs.get("query", ""),
                "llm_backend": "local-fallback",
                "workflow_status": "completed",
                "checkpoint": None,
                "provider_health": {},
                "result": result,
                "artifacts": {},
            }

        def controller() -> None:
            self.assertTrue(analyze_entered.wait(timeout=10))
            loop_holder["loop"].call_soon_threadsafe(lightweight_gate_holder["gate"].set)
            observed["lightweight_before_release"] = lightweight_done.wait(timeout=2)
            release_analyze.set()

        async def scenario() -> tuple[httpx.Response, httpx.Response]:
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                loop_holder["loop"] = asyncio.get_running_loop()
                lightweight_gate = asyncio.Event()
                lightweight_gate_holder["gate"] = lightweight_gate

                async def lightweight_request() -> httpx.Response:
                    await lightweight_gate.wait()
                    response = await client.get("/api/v1/config")
                    lightweight_done.set()
                    return response

                lightweight_task = asyncio.create_task(lightweight_request())
                upload_task = asyncio.create_task(
                    client.post(
                        "/api/v1/analyze-upload",
                        data={
                            "query": "Analyze Apple FY2025.",
                            "thread_id": "upload-concurrency",
                            "export_artifacts": "false",
                        },
                        files={"files": ("notes.txt", b"Apple revenue FY2025 was 100 billion.", "text/plain")},
                    )
                )
                lightweight = await lightweight_task
                upload = await upload_task
                return upload, lightweight

        controller_thread = Thread(target=controller, daemon=True)
        controller_thread.start()
        with patch.object(LumenFinAnalysisService, "analyze", blocked_analyze):
            upload_response, lightweight_response = asyncio.run(scenario())
        controller_thread.join(timeout=10)

        self.assertTrue(observed["lightweight_before_release"])
        self.assertEqual(lightweight_response.status_code, 200)
        self.assertEqual(upload_response.status_code, 200)

    def test_api_returns_409_for_checkpoint_conflicts(self) -> None:
        tmp_root = ROOT / "test_artifacts" / f"api-conflict-{uuid4().hex[:8]}"
        app = create_app(
            build_test_config(tmp_root),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        requests = [
            (
                "/api/v1/analyze",
                {"query": "Analyze Apple FY2025.", "thread_id": "conflict-thread"},
                "analyze",
            ),
            (
                "/api/v1/clarify",
                {
                    "thread_id": "conflict-thread",
                    "clarification": {"company": "Apple", "time_range": "FY2025"},
                },
                "clarify",
            ),
        ]
        with TestClient(app) as client:
            for path, payload, method_name in requests:
                with self.subTest(path=path), patch.object(
                    LumenFinAnalysisService,
                    method_name,
                    side_effect=CheckpointConflictError("stale checkpoint revision"),
                ):
                    response = client.post(path, json=payload)
                    self.assertEqual(response.status_code, 409)
                    self.assertIn("stale checkpoint revision", response.json()["detail"])

    def test_end_to_end_report_generation(self) -> None:
        app = build_offline_system()
        result = app.run("对比分析 Apple 与 Microsoft 2025 年供应链风险和研发投入。", thread_id="test-e2e-offline")

        self.assertIn("final_report", result)
        self.assertIn("Apple", result["final_report"])
        self.assertIn("Microsoft", result["final_report"])
        self.assertEqual(result["llm_backend"], "local-fallback")
        self.assertIn("Evidence Boundary", result["final_report"])
        self.assertIn("Risk-model scores remain screening indicators", result["final_report"])
        self.assertIn("Research Thesis & Positioning", result["final_report"])
        self.assertNotIn("Recommend overweight", result["final_report"])
        self.assertNotIn("cautious accumulation", result["final_report"])

        steps = [event["step"] for event in result["audit_log"]]
        for required in (
            "input_guardrail",
            "query_planner",
            "supervisor",
            "retrieval",
            "quant",
            "psychologist",
            "critic",
            "claim_binder",
            "synthesizer",
        ):
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

    def test_reused_system_starts_each_run_with_fresh_audit_log(self) -> None:
        app = build_offline_system()
        first = app.run("Analyze Apple FY2025 supply chain risk.", thread_id="audit-isolation-1")
        second = app.run("Analyze Tesla FY2025 liquidity risk.", thread_id="audit-isolation-2")

        expected_steps = [
            "input_guardrail",
            "query_planner",
            "supervisor",
            "retrieval",
            "quant",
            "psychologist",
            "critic",
            "claim_binder",
            "synthesizer",
        ]
        self.assertEqual(first["audit_log"][0]["step"], "input_guardrail")
        self.assertEqual([event["step"] for event in second["audit_log"]], expected_steps)

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
                    "include_state": "true",
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
