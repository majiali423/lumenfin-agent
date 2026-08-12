from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.reporting import (
    build_data_sources,
    build_run_manifest,
    format_rag_citation_section,
    humanize_citation,
    report_contains_page_citations,
)


class ReportingDataSourcesTestCase(unittest.TestCase):
    def test_sample_db_structured_source(self) -> None:
        sources = build_data_sources(
            {"companies": ["NVIDIA"], "document_contexts": [], "data_mode": "demo"},
            llm_backend="local-fallback",
        )
        self.assertEqual(sources["structured"], "sample_db")
        self.assertEqual(sources["data_mode"], "demo")
        self.assertEqual(sources["rag"], "skipped")
        self.assertFalse(sources["structured_uploaded"])

    def test_live_mode_does_not_label_sample_db(self) -> None:
        sources = build_data_sources(
            {"companies": ["NVIDIA"], "document_contexts": [], "data_mode": "live"},
            llm_backend="deepseek",
        )
        self.assertEqual(sources["structured"], "none")
        self.assertEqual(sources["data_mode"], "live")

    def test_uploaded_json_structured_source(self) -> None:
        sources = build_data_sources(
            {
                "companies": ["CustomCo"],
                "document_contexts": [
                    {
                        "filename": "CustomCo_metrics.json",
                        "source_type": "structured_json",
                    }
                ],
            }
        )
        self.assertEqual(sources["structured"], "uploaded_json")
        self.assertTrue(sources["structured_uploaded"])

    def test_rag_milvus_when_hits(self) -> None:
        sources = build_data_sources(
            {
                "companies": ["NVIDIA"],
                "document_contexts": [{"filename": "nvda.pdf"}],
                "rag_evidence": {"NVIDIA": [{"page": 1}]},
                "rag_index_stats": {"chunks_indexed": 3},
            },
            rag_enabled=True,
        )
        self.assertEqual(sources["rag"], "milvus_hybrid")
        self.assertTrue(sources["pdf_uploaded"])

    def test_market_ok_when_price_present(self) -> None:
        sources = build_data_sources(
            {
                "market_snapshots": {
                    "NVIDIA": {"provider": "yahoo", "current_price": 120.5}
                }
            },
            market_provider="yahoo",
        )
        self.assertTrue(sources["market_ok"])
        self.assertEqual(sources["market"], "yahoo")

    def test_uploaded_csv_structured_source(self) -> None:
        sources = build_data_sources(
            {
                "companies": ["NVIDIA"],
                "document_contexts": [
                    {
                        "filename": "nvidia_metrics.csv",
                        "source_type": "csv",
                        "metric_hints": {"revenue": 130.5},
                    }
                ],
            }
        )
        self.assertEqual(sources["structured"], "uploaded_csv")
        self.assertTrue(sources["structured_uploaded"])
        self.assertIn("csv", sources["upload_formats"])

    def test_manifest_includes_data_sources(self) -> None:
        manifest = build_run_manifest(
            {"companies": ["NVIDIA"], "workflow_status": "completed"},
            thread_id="t1",
            llm_backend="local-fallback",
        )
        self.assertIn("data_sources", manifest)
        self.assertEqual(manifest["data_sources"]["structured"], "sample_db")

    def test_humanize_citation_analyst_labels(self) -> None:
        self.assertEqual(humanize_citation("msft_10k.pdf#p12"), "msft_10k.pdf p.12")
        self.assertIn("SEC companyfacts", humanize_citation("lumenfin:sec_companyfacts:Apple:fy2024"))
        self.assertIn("Risk screening", humanize_citation("lumenfin:risk_model:Apple"))

    def test_format_rag_citation_section_emits_page_anchors(self) -> None:
        lines = format_rag_citation_section(
            {
                "Apple": [
                    {
                        "citation": "apple_msft_fy2025_table.pdf#p1",
                        "retrieval_method": "hybrid",
                        "text": "Apple revenue $391B",
                    }
                ],
                "Microsoft": [
                    {
                        "filename": "apple_msft_fy2025_table.pdf",
                        "page": 1,
                        "method": "dense",
                        "excerpt": "Microsoft revenue $245B",
                    }
                ],
            }
        )
        report = "\n".join(lines)
        self.assertIn("Retrieved Document Citations", report)
        self.assertIn("apple_msft_fy2025_table.pdf p.1", report)
        self.assertTrue(report_contains_page_citations(report))
        self.assertEqual(format_rag_citation_section({}), [])
        self.assertFalse(report_contains_page_citations("no citations here"))

        # Dedupes near-identical excerpts; prefers higher-score hits.
        deduped = format_rag_citation_section(
            {
                "Apple": [
                    {
                        "citation": "a.pdf#p1",
                        "score": 0.2,
                        "text": "Same revenue excerpt for Apple filing table",
                    },
                    {
                        "citation": "a.pdf#p2",
                        "score": 0.9,
                        "text": "Same revenue excerpt for Apple filing table",
                    },
                    {
                        "citation": "a.pdf#p3",
                        "score": 0.8,
                        "text": "Distinct R&D intensity note for Apple",
                    },
                ]
            },
            max_rows_per_company=2,
        )
        dedupe_report = "\n".join(deduped)
        self.assertEqual(dedupe_report.count("| Apple |"), 2)
        self.assertIn("Distinct R&D", dedupe_report)

    def test_rag_citations_preserve_rerank_order_over_retrieval_score(self) -> None:
        report = "\n".join(
            format_rag_citation_section(
                {
                    "Apple": [
                        {
                            "citation": "wrong-period.pdf#p1",
                            "score": 0.99,
                            "rerank_score": 0.1,
                            "text": "Apple FY2024 revenue distractor",
                        },
                        {
                            "citation": "correct-period.pdf#p2",
                            "score": 0.2,
                            "rerank_score": 0.95,
                            "text": "Apple FY2025 revenue answer",
                        },
                    ]
                },
                max_rows_per_company=2,
            )
        )

        self.assertLess(report.index("correct-period.pdf"), report.index("wrong-period.pdf"))


if __name__ == "__main__":
    unittest.main()
