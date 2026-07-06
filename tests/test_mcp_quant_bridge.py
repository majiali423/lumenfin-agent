from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.tools import resolve_safe_formula, safe_execute_formula


class McpQuantBridgeTestCase(unittest.TestCase):
    def test_local_backend_matches_core(self) -> None:
        variables = {"ebitda": 141.2, "revenue": 412.0}
        formula = "ebitda / revenue"
        self.assertEqual(
            resolve_safe_formula(formula, variables, backend="local"),
            safe_execute_formula(formula, variables),
        )

    def test_mcp_backend_delegates_to_bridge(self) -> None:
        variables = {"ebitda": 141.2, "revenue": 412.0}
        formula = "ebitda / revenue"
        with patch("lumenfin.mcp_bridge.compute_ratio_via_mcp", return_value=0.3422) as mocked:
            value = resolve_safe_formula(formula, variables, backend="mcp")
        mocked.assert_called_once_with(formula, variables)
        self.assertEqual(value, 0.3422)


if __name__ == "__main__":
    unittest.main()
