from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from lumenfin.api.app import create_app
from lumenfin.config import AppConfig
from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.tools import retrieve_company_payload
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class ProductionMindsetThinGuardsTestCase(unittest.TestCase):
    def test_production_env_defaults_to_live_data_and_no_local_fallback(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "production",
                "MAS_API_KEY": "prod-key",
                "DEEPSEEK_API_KEY": "",
            },
            clear=True,
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.app_env, "production")
        self.assertEqual(config.data_mode, "live")
        self.assertFalse(config.allows_sample_data())
        self.assertFalse(config.allows_local_fallback())

    def test_production_demo_and_fallback_require_explicit_opt_in(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "APP_ENV": "production",
                "MAS_API_KEY": "prod-key",
                "DATA_MODE": "demo",
                "ALLOW_LOCAL_FALLBACK": "true",
            },
            clear=True,
        ):
            config = AppConfig.from_env()

        self.assertEqual(config.data_mode, "demo")
        self.assertTrue(config.allows_sample_data())
        self.assertTrue(config.allows_local_fallback())

    def test_live_mode_skips_sample_financial_payload(self) -> None:
        demo = retrieve_company_payload("Apple", allow_sample_data=True)
        live = retrieve_company_payload("Apple", allow_sample_data=False)
        self.assertEqual(demo.get("structured_source"), "sample_db")
        self.assertNotEqual(live.get("structured_source"), "sample_db")
        self.assertFalse(live.get("market_data"))

    def test_upload_rejects_oversized_and_bad_extension(self) -> None:
        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"upload-{uuid4().hex[:8]}"),
            max_upload_bytes=16,
            max_upload_files=1,
        )
        from lumenfin.service import LumenFinAnalysisService

        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        with self.assertRaises(ValueError):
            service.save_uploaded_files([("notes.exe", b"not-allowed")])
        with self.assertRaises(ValueError):
            service.save_uploaded_files([("big.pdf", b"x" * 64)])

    def test_create_app_requires_api_key_outside_dev(self) -> None:
        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"auth-{uuid4().hex[:8]}"),
            app_env="production",
            api_key=None,
        )
        with self.assertRaises(RuntimeError):
            create_app(
                config,
                llm_client=LocalFallbackLLMClient(),
                market_data_client=FakeMarketDataClient(),
            )

    def test_analyze_response_defaults_to_compact_state(self) -> None:
        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"api-{uuid4().hex[:8]}"),
            api_key="test-key",
            app_env="test",
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        client = TestClient(app)
        response = client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": "test-key"},
            json={"query": "Analyze Apple FY2025 supply chain risk.", "export_artifacts": False},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("data_mode", payload["state"])
        self.assertNotIn("retrieved_docs", payload["state"])


if __name__ == "__main__":
    unittest.main()
