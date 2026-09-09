from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "static" / "index.html"
APP_JS = ROOT / "static" / "app.js"


class Phase5DemoUiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = INDEX.read_text(encoding="utf-8")
        cls.js = APP_JS.read_text(encoding="utf-8")
        cls.frontend = cls.html + "\n" + cls.js

    def test_no_simulated_800ms_progress(self) -> None:
        self.assertNotIn("progressTimer", self.frontend)
        self.assertNotIn(",800)", self.js)
        self.assertIn("pollJob", self.js)
        self.assertIn("/api/v1/jobs", self.js)
        self.assertIn("?job=", self.js)
        self.assertIn("job_id: activeJobId", self.js)
        self.assertIn("needs_clarification", self.js)

    def test_core_page_does_not_require_external_cdn(self) -> None:
        self.assertNotIn("cdn.jsdelivr.net", self.html)
        self.assertNotIn("unpkg.com", self.html)
        self.assertIn('src="/static/sanitize.js"', self.html)
        self.assertIn('src="/static/app.js"', self.html)

    def test_default_tab_is_concise_answer(self) -> None:
        self.assertIn('data-tab="answer"', self.html)
        self.assertIn("简洁回答", self.html)
        self.assertIn("id=\"formulaTable\"", self.html)
        self.assertIn("id=\"evidenceList\"", self.html)
        self.assertIn('createElement("thead")', self.js)
        self.assertIn('createElement("tbody")', self.js)
        self.assertNotIn("<thead><tr><th>公司</th>", self.js)

    def test_bench_drawer_is_a_mutation_walkthrough_not_product_accuracy(self) -> None:
        self.assertIn("干净 FinRun 通过", self.html)
        self.assertIn("999999%", self.html)
        self.assertIn("不代表通用金融准确率", self.html)
        self.assertIn("离线 Case 合同分", self.html)
        self.assertIn("不是当前页面这次分析的实时得分", self.html)
        self.assertNotIn("PASS · 100", self.html)


class Phase5JobPollApiTestCase(unittest.TestCase):
    def test_job_poll_exposes_audit_without_retrieved_docs(self) -> None:
        from dataclasses import replace

        from lumenfin.api.app import create_app
        from lumenfin.database import JobRepository
        from lumenfin.llm import LocalFallbackLLMClient
        from tests.support.fakes import FakeMarketDataClient
        from tests.test_graph_routing import build_test_config

        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"p5-job-{uuid4().hex[:8]}"),
            api_key="test-key",
            rag_enabled=False,
        )
        repo = JobRepository(config.database_url, db_path=config.db_path)
        repo.create_job("job-p5", "thread-p5", "Analyze Apple FY2025.", tenant_id="test-tenant")
        repo.update_job_status(
            "job-p5",
            status="completed",
            result={
                "thread_id": "thread-p5",
                "workflow_status": "completed",
                "final_report": "Apple revenue 412.0",
                "answer": "Apple sample revenue is 412.0.",
                "citations": ["chunk-1"],
                "audit_log": [{"step": "retrieval", "status": "ok", "detail": "done"}],
                "financial_metrics": {"Apple": {"ebitda_margin": 0.34}},
                "retrieved_docs": {"Apple": {"secret": "do-not-leak"}},
            },
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        payload = TestClient(app).get("/api/v1/jobs/job-p5", headers={"X-API-Key": "test-key"}).json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["audit_log"][0]["step"], "retrieval")
        self.assertEqual(payload["result"]["answer"], "Apple sample revenue is 412.0.")
        self.assertIn("ebitda_margin", payload["result"]["financial_metrics"]["Apple"])
        self.assertNotIn("retrieved_docs", payload["result"])
        self.assertNotIn("do-not-leak", str(payload))

    def test_static_assets_are_served(self) -> None:
        from dataclasses import replace

        from lumenfin.api.app import create_app
        from lumenfin.llm import LocalFallbackLLMClient
        from tests.support.fakes import FakeMarketDataClient
        from tests.test_graph_routing import build_test_config

        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"p5-static-{uuid4().hex[:8]}"),
            api_key=None,
            rag_enabled=False,
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        client = TestClient(app)
        index = client.get("/static/index.html")
        js = client.get("/static/app.js")
        self.assertEqual(index.status_code, 200)
        self.assertEqual(js.status_code, 200)
        self.assertIn("pollJob", js.text)
        self.assertNotIn("progressTimer", index.text + js.text)

    def test_config_exposes_data_mode(self) -> None:
        from dataclasses import replace

        from lumenfin.api.app import create_app
        from lumenfin.llm import LocalFallbackLLMClient
        from tests.support.fakes import FakeMarketDataClient
        from tests.test_graph_routing import build_test_config

        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"p5-cfg-{uuid4().hex[:8]}"),
            api_key="test-key",
            rag_enabled=False,
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        body = TestClient(app).get("/api/v1/config", headers={"X-API-Key": "test-key"}).json()
        self.assertEqual(body["data_mode"], "demo")


if __name__ == "__main__":
    unittest.main()
