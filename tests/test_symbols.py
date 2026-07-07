from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.tools import derive_target_symbols


class SymbolDerivationTestCase(unittest.TestCase):
    def test_uses_default_ticker_map_without_generic_uppercase_capture(self) -> None:
        companies = ["NVIDIA", "AMD"]
        query = "Compare NVIDIA and AMD FY2025 R&D intensity"
        symbols = derive_target_symbols(companies, query)
        self.assertEqual(symbols["NVIDIA"], "NVDA")
        self.assertEqual(symbols["AMD"], "AMD")

    def test_allows_explicit_ticker_override(self) -> None:
        companies = ["NVIDIA"]
        query = "Analyze NVIDIA with ticker: NVDA"
        symbols = derive_target_symbols(companies, query)
        self.assertEqual(symbols["NVIDIA"], "NVDA")


if __name__ == "__main__":
    unittest.main()
