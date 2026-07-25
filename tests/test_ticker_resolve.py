from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.ticker_resolve import (
    enrich_company_universe,
    resolve_ticker_for_company,
    set_ticker_directory_for_tests,
)
from lumenfin.tools import derive_target_symbols


class TickerResolveTestCase(unittest.TestCase):
    def setUp(self) -> None:
        set_ticker_directory_for_tests(
            [
                {"ticker": "COST", "title": "COSTCO WHOLESALE CORP /NEW", "cik_str": 909832},
                {"ticker": "CRWD", "title": "CrowdStrike Holdings, Inc.", "cik_str": 1535527},
                {"ticker": "AAPL", "title": "Apple Inc.", "cik_str": 320193},
            ]
        )

    def tearDown(self) -> None:
        set_ticker_directory_for_tests(None)

    def test_default_map_still_wins_for_known_brands(self) -> None:
        hit = resolve_ticker_for_company("Apple", allow_network=False)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.ticker, "AAPL")
        self.assertEqual(hit.source, "default_map")

    def test_resolves_bare_sec_ticker_token(self) -> None:
        hit = resolve_ticker_for_company("COST", allow_network=True)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.ticker, "COST")
        self.assertEqual(hit.source, "sec_ticker")
        self.assertIn("Costco", hit.company)

    def test_resolves_company_name_via_sec_title(self) -> None:
        hit = resolve_ticker_for_company("Costco", allow_network=True)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.ticker, "COST")
        self.assertEqual(hit.source, "sec_title")

    def test_enrich_picks_up_query_ticker(self) -> None:
        companies, symbols, notes = enrich_company_universe(
            [],
            query="Diligence on CRWD FY2024 margins",
            allow_network=True,
        )
        self.assertTrue(any("Crowd" in c or c == "Crowdstrike" for c in companies) or "CRWD" in symbols.values())
        self.assertIn("CRWD", {s.upper() for s in symbols.values()})
        self.assertTrue(any("CRWD" in n for n in notes))

    def test_enrich_ignores_lowercase_live_word(self) -> None:
        set_ticker_directory_for_tests(
            [
                {"ticker": "LIVE", "title": "LIVE VENTURES Inc", "cik_str": 1045742},
                {"ticker": "AAPL", "title": "Apple Inc.", "cik_str": 320193},
            ]
        )
        companies, symbols, _notes = enrich_company_universe(
            ["OpenAI"],
            query="Analyze OpenAI FY2025 profitability using live fundamentals only.",
            allow_network=True,
        )
        self.assertEqual(companies, ["OpenAI"])
        self.assertNotIn("LIVE", {s.upper() for s in symbols.values()})

    def test_derive_target_symbols_uses_sec_directory(self) -> None:
        symbols = derive_target_symbols(["Costco"], "Analyze Costco profitability")
        self.assertEqual(symbols.get("Costco"), "COST")


if __name__ == "__main__":
    unittest.main()
