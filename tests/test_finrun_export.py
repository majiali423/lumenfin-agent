from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.finrun import export_finrun_state


class FinRunExportTestCase(unittest.TestCase):
    def test_export_finrun_state_maps_lumenfin_trace(self) -> None:
        finrun = export_finrun_state(_sample_state())

        self.assertEqual(finrun["run_id"], "lumenfin-sample")
        self.assertEqual({entity["name"] for entity in finrun["entities"]}, {"Apple"})
        self.assertTrue(any(step["name"] == "retrieval" for step in finrun["steps"]))
        self.assertTrue(any(metric["name"] == "ebitda_margin" for metric in finrun["metrics"]))
        self.assertTrue(any(item["source_type"] == "sample_db" for item in finrun["evidence"]))
        self.assertEqual(finrun["market_data"][0]["status"], "ok")

    def test_export_finrun_script_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "state.json"
            out_path = Path(tmp) / "finrun.json"
            state_path.write_text(json.dumps(_sample_state()), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_finrun.py"),
                    str(state_path),
                    "--out",
                    str(out_path),
                ],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("WROTE", completed.stdout)
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_id"], "lumenfin-sample")


def _sample_state() -> dict:
    return {
        "thread_id": "lumenfin-sample",
        "query": "Compare Apple FY2025 financial performance and risk.",
        "workflow_status": "completed",
        "llm_backend": "local-fallback",
        "companies": ["Apple"],
        "audit_log": [
            {"step": "query_planner", "status": "ok"},
            {"step": "retrieval", "status": "ok"},
            {"step": "quant", "status": "ok"},
            {"step": "synthesizer", "status": "ok"},
        ],
        "retrieved_docs": {
            "Apple": {
                "market_data": {
                    "revenue_2025": 412.0,
                    "ebitda_2025": 141.2,
                    "r_and_d_2025": 33.4,
                    "operating_income_2025": 123.6,
                },
                "source_documents": [
                    {"filename": "apple_2025.md", "excerpt": "Apple reported FY2025 revenue of 412.0 billion USD."}
                ],
            }
        },
        "financial_metrics": {"Apple": {"ebitda_margin": 0.3427, "r_and_d_intensity": 0.0811}},
        "market_snapshots": {"Apple": {"provider": "fake", "status": "ok", "current_price": 180.0}},
        "final_report": "## 1. Executive Summary\nApple analysis.\n\n## 4. Financial Performance Analysis\nEBITDA margin was 34.27%.\n\n## Risk\nMarket risk and data limitation apply.\n\n## Compliance\nResearch output only.\n\n## Methodology\nLumenFin trace.\n\n**Disclaimer:** Not investment advice.",
    }


if __name__ == "__main__":
    unittest.main()
