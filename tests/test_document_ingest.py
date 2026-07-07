from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.document_ingest import (  # noqa: E402
    parse_csv_documents,
    parse_json_documents,
    parse_markdown_document,
    parse_upload_documents,
)


class DocumentIngestTestCase(unittest.TestCase):
    def test_parse_csv_with_company_column(self) -> None:
        csv_text = "company,revenue_2025,ebitda_2025\nNVIDIA,130.5,75.2\nAMD,25.1,5.3\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.csv"
            path.write_text(csv_text, encoding="utf-8")
            contexts = parse_csv_documents(path)

        self.assertEqual(len(contexts), 2)
        by_company = {ctx["detected_companies"][0]: ctx for ctx in contexts}
        self.assertEqual(by_company["NVIDIA"]["source_type"], "csv")
        self.assertEqual(by_company["NVIDIA"]["metric_hints"]["revenue"], 130.5)
        self.assertEqual(by_company["AMD"]["metric_hints"]["ebitda"], 5.3)

    def test_parse_csv_single_company_from_filename(self) -> None:
        csv_text = "revenue_2025,ebitda_2025\n100.0,40.0\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "NVIDIA.csv"
            path.write_text(csv_text, encoding="utf-8")
            contexts = parse_csv_documents(path)

        self.assertEqual(len(contexts), 1)
        ctx = contexts[0]
        self.assertEqual(ctx["detected_companies"], ["NVIDIA"])
        self.assertEqual(ctx["metric_hints"]["revenue"], 100.0)

    def test_parse_markdown_sections(self) -> None:
        md = "# NVIDIA FY2025\n\nRevenue reached 130.5 billion.\n\n## Risks\nSupply chain risk noted.\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nvidia.md"
            path.write_text(md, encoding="utf-8")
            ctx = parse_markdown_document(path)

        self.assertEqual(ctx["source_type"], "markdown")
        self.assertEqual(len(ctx["pages"]), 2)
        self.assertIn("NVIDIA", ctx["detected_companies"])

    def test_parse_json_upload(self) -> None:
        payload = {"NVIDIA": {"revenue_2025": 88.0, "ebitda_2025": 44.0}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            contexts = parse_json_documents(path)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["source_type"], "structured_json")
        self.assertEqual(contexts[0]["metric_hints"]["revenue"], 88.0)
        self.assertEqual(contexts[0]["path"], str(path))

    def test_parse_upload_router(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "note.markdown"
            md_path.write_text("## Tencent\nTencent revenue discussion.", encoding="utf-8")
            contexts = parse_upload_documents(md_path)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["source_type"], "markdown")

    def test_parse_excel_sheet(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.xlsx"
            wb = Workbook()
            ws = wb.active
            ws.title = "NVIDIA"
            ws.append(["metric", "value"])
            ws.append(["revenue_2025", 120.0])
            ws.append(["ebitda_2025", 60.0])
            wb.save(path)
            contexts = parse_upload_documents(path)

        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["source_type"], "excel")
        self.assertEqual(contexts[0]["metric_hints"]["revenue"], 120.0)

    def test_unsupported_suffix_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "chart.png"
            path.write_bytes(b"fake")
            with self.assertRaises(ValueError):
                parse_upload_documents(path)


if __name__ == "__main__":
    unittest.main()
