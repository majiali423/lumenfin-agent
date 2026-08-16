from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lumenfin.eval.financebench.fetch_pdfs import candidate_urls
from lumenfin.eval.financebench.reporting import (
    aggregate_case_metrics,
    assert_no_secrets,
    compare_modes,
    redact_mapping,
    write_json,
)
from lumenfin.eval.financebench.split import SplitError, forbid_test_split_tuning


class FinanceBenchReportingTestCase(unittest.TestCase):
    def test_redacts_keys_and_urls(self) -> None:
        payload = redact_mapping(
            {
                "DASHSCOPE_API_KEY": "sk-abcdefghijklmnopqrstuvwxyz012345",
                "endpoint": "https://dashscope.aliyuncs.com/compatible-api/v1",
                "ok": "page_mrr",
            }
        )
        self.assertEqual(payload["DASHSCOPE_API_KEY"], "[REDACTED]")
        self.assertEqual(payload["endpoint"], "[REDACTED]")
        self.assertEqual(payload["ok"], "page_mrr")
        with self.assertRaises(ValueError):
            assert_no_secrets({"token": "sk-abcdefghijklmnopqrstuvwxyz012345"})

    def test_write_json_redacts_before_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "env.json"
            write_json(path, {"authorization": "Bearer secret-token", "mode": "bm25"})
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["authorization"], "[REDACTED]")
            self.assertEqual(loaded["mode"], "bm25")

    def test_compare_modes_tracks_rank_movement(self) -> None:
        def row(case_id: str, rank: int, hit: float) -> dict:
            return {
                "case_id": case_id,
                "status": "ok",
                "single_gold_page": True,
                "labels": {"question_type": "metrics-generated"},
                "page": {
                    "hit_at": {"1": hit, "3": hit, "5": hit, "10": hit},
                    "recall_at": {"1": hit, "3": hit, "5": hit, "10": hit},
                    "mrr": hit,
                    "ndcg_at": {"5": hit, "10": hit},
                    "first_relevant_rank": rank,
                },
                "chunk": {
                    "hit_at": {"5": hit, "10": hit, "20": hit},
                    "recall_at": {"5": hit, "10": hit, "20": hit},
                    "mrr": hit,
                    "ndcg_at": {"10": hit},
                },
            }

        comparison = compare_modes(
            {
                "bm25": [row("fb-a", 5, 1.0), row("fb-b", 0, 0.0)],
                "hybrid": [row("fb-a", 1, 1.0), row("fb-b", 0, 0.0)],
            }
        )
        self.assertEqual(comparison["improved"], 1)
        self.assertEqual(comparison["never_retrieved"], 1)
        summary = aggregate_case_metrics(comparison["aggregates"]["bm25"] and [row("fb-a", 5, 1.0)])
        self.assertEqual(summary["cases"], 1)

    def test_adobe_and_sec_candidate_urls(self) -> None:
        adobe = (
            "https://www.adobe.com/pdf-page.html?pdfTarget="
            "aHR0cHM6Ly93d3cuYWRvYmUuY29tL2ZpbGUucGRm"
        )
        urls = candidate_urls(adobe)
        self.assertTrue(any(item.endswith("file.pdf") for item in urls))
        secish = (
            "https://investors.3m.com/financials/sec-filings/content/"
            "0000066740-23-000014/0000066740-23-000014.pdf"
        )
        sec_urls = candidate_urls(secish)
        self.assertTrue(any("sec.gov/Archives/edgar/data/66740/" in item for item in sec_urls))

    def test_cli_tune_flag_rejected_on_test(self) -> None:
        with self.assertRaises(SplitError):
            forbid_test_split_tuning("test", tuning=True)


if __name__ == "__main__":
    unittest.main()
