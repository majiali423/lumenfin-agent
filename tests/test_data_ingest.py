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

from lumenfin.data_ingest import (
    load_metrics_json_file,
    normalize_metric_hints,
    structured_metrics_to_document_contexts,
)


class DataIngestTestCase(unittest.TestCase):
    def test_normalize_metric_hints_maps_year_suffix(self) -> None:
        hints = normalize_metric_hints(
            {"revenue_2025": "130.5", "ebitda_2025": 75.2, "unknown": 1}
        )
        self.assertEqual(hints["revenue"], 130.5)
        self.assertEqual(hints["ebitda"], 75.2)
        self.assertNotIn("unknown", hints)

    def test_structured_metrics_to_document_contexts(self) -> None:
        contexts = structured_metrics_to_document_contexts(
            {"NVIDIA": {"revenue_2025": 100.0, "ebitda_2025": 40.0}}
        )
        self.assertEqual(len(contexts), 1)
        ctx = contexts[0]
        self.assertEqual(ctx["source_type"], "structured_json")
        self.assertEqual(ctx["detected_companies"], ["NVIDIA"])
        self.assertEqual(ctx["metric_hints"]["revenue"], 100.0)
        self.assertTrue(ctx["filename"].endswith("_metrics.json"))

    def test_load_metrics_json_file(self) -> None:
        payload = {"Apple": {"revenue_2025": 1.0}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_metrics_json_file(path)
        self.assertEqual(loaded, payload)

    def test_load_metrics_json_file_rejects_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_metrics_json_file(path)


if __name__ == "__main__":
    unittest.main()
