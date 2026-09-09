"""Exporter vs FinAgentBench LumenFin adapter: comparable fields and schema."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from tests.support.finagentbench import finagentbench_root

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = finagentbench_root()
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from lumenfin.finrun import export_finrun_state


class FinRunAdapterParityTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sample_path = BENCH / "fixtures" / "lumenfin_state_sample.json"
        cls.state = json.loads(sample_path.read_text(encoding="utf-8"))

    def test_exporter_and_adapter_share_normalized_visible_fields(self) -> None:
        from finagentbench.adapters.compare import comparable_finrun
        from finagentbench.adapters.lumenfin import LumenFinAdapter
        from finagentbench.schema import validate_finrun

        exported = export_finrun_state(self.state)
        adapted = LumenFinAdapter().normalize(self.state)
        validate_finrun(exported)
        validate_finrun(adapted)
        left = comparable_finrun(exported)
        right = comparable_finrun(adapted)
        self.assertEqual(left["entities"], right["entities"])
        self.assertEqual(left["final_output"], right["final_output"])
        self.assertEqual(left["metrics"], right["metrics"])
        self.assertEqual(left["steps"], right["steps"])
        self.assertTrue(exported["final_output"])
        self.assertEqual(exported["metadata"].get("execution_path"), "product_workflow")
        self.assertEqual(adapted["metadata"].get("execution_path"), "contract_replay")


if __name__ == "__main__":
    unittest.main()
