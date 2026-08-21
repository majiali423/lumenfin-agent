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

from lumenfin.finrun import FINRUN_SCHEMA_VERSION, export_finrun_state
from lumenfin.structured_answer import (
    CITATION_PATH_VALIDATION_FAILED,
    CITATION_VALIDATION_FAILED,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
)


class FinRunExportTestCase(unittest.TestCase):
    def test_export_finrun_state_maps_lumenfin_trace(self) -> None:
        finrun = export_finrun_state(_sample_state())

        self.assertEqual(finrun["schema_version"], FINRUN_SCHEMA_VERSION)
        self.assertEqual(finrun["run_id"], "lumenfin-sample")
        self.assertEqual({entity["name"] for entity in finrun["entities"]}, {"Apple"})
        self.assertTrue(any(step["name"] == "retrieval" for step in finrun["steps"]))
        self.assertTrue(any(metric["name"] == "ebitda_margin" for metric in finrun["metrics"]))
        self.assertTrue(any(item["source_type"] == "sample_db" for item in finrun["evidence"]))
        self.assertTrue(any(item["source_type"] == "risk_model" for item in finrun["evidence"]))
        self.assertTrue(any(item["source_type"] == "market_data" for item in finrun["evidence"]))
        self.assertEqual(finrun["market_data"][0]["status"], "ok")
        self.assertEqual(finrun["market_data"][0]["current_price"], 180.0)
        self.assertEqual(finrun["metadata"]["data_mode"], "demo")
        self.assertEqual(finrun["metadata"]["compliance_violations"], [])
        self.assertEqual(
            finrun["metadata"]["retrieval_provenance"]["Apple"]["structured_source"],
            "sample_db",
        )
        metric = next(item for item in finrun["metrics"] if item["name"] == "ebitda_margin")
        self.assertEqual(metric["confidence"]["structured_source"], "sample_db")
        self.assertEqual(metric["inputs"]["revenue"]["period_source"], "provider_record")
        self.assertEqual(metric["inputs"]["revenue"]["source_record_id"], "sample:apple:FY2025:revenue")
        structured = finrun["structured_answer"]
        self.assertEqual(finrun["schema_version"], FINRUN_SCHEMA_VERSION)
        self.assertEqual(structured["structured_answer_schema_version"], STRUCTURED_ANSWER_SCHEMA_VERSION)
        self.assertEqual(structured["citations"], [])
        self.assertEqual(structured["citation_source"], "unavailable")
        self.assertEqual(structured["citation_validation"], "passed")
        self.assertEqual(structured["citation_path"], "unavailable")
        self.assertEqual(finrun["metadata"]["citation_validation"], "passed")

    def test_export_preserves_structured_citations_and_does_not_guess_from_prose(self) -> None:
        state = _sample_state()
        state["rag_evidence"] = {
            "Apple": [
                {
                    "chunk_id": "apple:p1:c0",
                    "citation": "10k.pdf#p1",
                    "text": "Apple reported FY2025 revenue of 412.0 billion USD.",
                    "tenant_id": "tenant-a",
                    "session_id": "lumenfin-sample",
                }
            ]
        }
        state["rag_tenant_id"] = "tenant-a"
        state["verified_claims"] = [
            {
                "claim_id": "c1",
                "entity": "Apple",
                "claim_type": "numeric",
                "statement": "Apple revenue was 412.",
                "verification": "verified",
                "evidence_refs": [
                    {
                        "evidence_id": "ev1",
                        "entity": "Apple",
                        "citation": "10k.pdf#p1",
                        "source_type": "rag",
                        "text": "Apple reported FY2025 revenue of 412.0 billion USD.",
                        "chunk_id": "apple:p1:c0",
                        "tenant_id": "tenant-a",
                        "session_id": "lumenfin-sample",
                    }
                ],
            }
        ]
        state["claims"] = state["verified_claims"]
        finrun = export_finrun_state(state)
        self.assertEqual(finrun["structured_answer"]["citations"], ["apple:p1:c0"])
        self.assertEqual(finrun["structured_answer"]["citation_source"], "structured")
        self.assertEqual(finrun["structured_answer"]["citation_validation"], "passed")
        self.assertTrue(any(item.get("chunk_id") == "apple:p1:c0" for item in finrun["evidence"]))
        self.assertEqual(finrun["metadata"]["citation_path"], "verified_evidence.chunk_id")

    def test_invalid_model_invented_chunk_is_not_exported(self) -> None:
        state = _sample_state()
        state["structured_answer"] = {
            "answer": state["final_report"],
            "citations": ["invented-chunk"],
            "structured_answer_schema_version": "1.0",
            "citation_source": "structured",
        }
        # Pre-attached invalid object is still serialized, but builder path
        # used when structured_answer lacks schema is fail-closed. Explicit
        # unknown IDs must not be treated as valid evidence rows.
        finrun = export_finrun_state(state)
        self.assertEqual(finrun["structured_answer"]["citations"], [])
        self.assertEqual(finrun["structured_answer"]["citation_source"], "unavailable")
        self.assertEqual(
            finrun["structured_answer"]["citation_validation"],
            CITATION_VALIDATION_FAILED,
        )
        self.assertEqual(
            finrun["structured_answer"]["citation_path"],
            CITATION_PATH_VALIDATION_FAILED,
        )
        self.assertEqual(finrun["metadata"]["citation_validation"], CITATION_VALIDATION_FAILED)
        self.assertEqual(finrun["metadata"]["citation_path"], CITATION_PATH_VALIDATION_FAILED)
        error = str(finrun["structured_answer"].get("validation_error") or "")
        self.assertTrue(error)
        self.assertNotIn("invented-chunk", error)
        self.assertNotIn("412.0 billion", error)
        self.assertFalse(any(item.get("chunk_id") == "invented-chunk" for item in finrun["evidence"]))

    def test_unverified_and_cross_scope_citations_degrade_explicitly(self) -> None:
        state = _sample_state()
        state["rag_tenant_id"] = "tenant-a"
        state["rag_evidence"] = {
            "Apple": [
                {
                    "chunk_id": "apple:p1:c0",
                    "citation": "10k.pdf#p1",
                    "text": "Apple revenue.",
                    "tenant_id": "tenant-b",
                    "session_id": "lumenfin-sample",
                }
            ]
        }
        state["verified_claims"] = [
            {
                "claim_id": "c1",
                "entity": "Apple",
                "claim_type": "numeric",
                "statement": "Apple revenue was 412.",
                "verification": "verified",
                "value": 412.0,
                "evidence_refs": [
                    {
                        "evidence_id": "ev1",
                        "entity": "Apple",
                        "citation": "10k.pdf#p1",
                        "source_type": "rag",
                        "text": "Apple revenue.",
                        "chunk_id": "apple:p1:c0",
                        "tenant_id": "tenant-b",
                        "session_id": "lumenfin-sample",
                    }
                ],
            }
        ]
        state["claims"] = state["verified_claims"]
        finrun = export_finrun_state(state)
        self.assertEqual(finrun["structured_answer"]["citation_validation"], CITATION_VALIDATION_FAILED)
        self.assertEqual(finrun["structured_answer"]["citation_source"], "unavailable")
        self.assertEqual(finrun["structured_answer"]["citations"], [])

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
                "supply_chain": {
                    "risk_level": "medium",
                    "signals": ["Supplier concentration remains above target."],
                },
                "earnings_call_quotes": ["Management cited services expansion and margin discipline."],
                "structured_source": "sample_db",
                "fundamentals_meta": {"fiscal_year": 2025},
                "fundamental_provenance": {
                    key: {
                        "source": "provider_record", "confidence": "high",
                        "period": "FY2025", "period_source": "provider_record",
                        "period_alignment": "exact", "citation": f"sample://apple/{key}",
                        "source_record_id": f"sample:apple:FY2025:{key}",
                    }
                    for key in ("revenue", "ebitda", "r_and_d", "operating_income")
                },
                "provenance": {
                    "structured_source": "sample_db",
                    "market_provider": "fake",
                    "market_status": "ok",
                    "data_mode": "demo",
                },
                "confidence": {"overall": 0.85, "market_data": 1.0, "live_market": 1.0, "rag_coverage": 0.0},
            }
        },
        "data_mode": "demo",
        "input_guardrail_summary": {"allowed": True, "mode": "sanitize", "finding_count": 0, "critical_count": 0},
        "compliance_violations": [],
        "retrieval_provenance": {
            "Apple": {"structured_source": "sample_db", "market_provider": "fake", "market_status": "ok", "data_mode": "demo"}
        },
        "financial_metrics": {"Apple": {"ebitda_margin": 0.3427, "r_and_d_intensity": 0.0811}},
        "risk_scores": {"Apple": {"supply_chain_risk": 5.0, "market_risk": 4.0}},
        "market_snapshots": {"Apple": {"provider": "fake", "status": "ok", "current_price": 180.0}},
        "final_report": "## 1. Executive Summary\nApple analysis.\n\n## 4. Financial Performance Analysis\nEBITDA margin was 34.27%.\n\n## Risk\nMarket risk and data limitation apply.\n\n## Compliance\nResearch output only.\n\n## Methodology\nLumenFin trace.\n\n**Disclaimer:** Not investment advice.",
    }


if __name__ == "__main__":
    unittest.main()
