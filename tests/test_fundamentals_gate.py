from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.tools import (
    build_coverage_matrix,
    canonicalize_companies,
    classify_quant_status,
    has_computable_fundamentals,
    is_partial_compare_gap,
    non_comparable_companies,
)


class FundamentalsGateTests(unittest.TestCase):
    def test_canonicalize_tencent_aliases(self) -> None:
        self.assertEqual(
            canonicalize_companies(["腾讯", "腾讯控股", "Tencent", "tencent"]),
            ["Tencent"],
        )

    def test_has_computable_requires_revenue_and_peer_input(self) -> None:
        self.assertFalse(has_computable_fundamentals(None))
        self.assertFalse(has_computable_fundamentals({}))
        self.assertFalse(has_computable_fundamentals({"market_data": {"revenue_2025": 100}}))
        self.assertTrue(
            has_computable_fundamentals(
                {"market_data": {"revenue_2025": 100, "ebitda_2025": 40}}
            )
        )

    def test_classify_quant_status(self) -> None:
        self.assertEqual(
            classify_quant_status({"ebitda_margin": 0.3, "current_price": 100.0}),
            "ast_ok",
        )
        self.assertEqual(classify_quant_status({"current_price": 100.0}), "market_only")
        self.assertEqual(classify_quant_status({}), "uncomputable")

    def test_non_comparable_list(self) -> None:
        matrix = build_coverage_matrix(
            ["Apple", "Microsoft"],
            {
                "Apple": {
                    "structured_source": "sec_companyfacts",
                    "market_data": {"revenue_2025": 100, "ebitda_2025": 40},
                },
                "Microsoft": {
                    "structured_source": "none",
                    "market_data": {},
                },
            },
        )
        self.assertTrue(is_partial_compare_gap(["Apple", "Microsoft"], matrix))
        self.assertEqual(non_comparable_companies(["Apple", "Microsoft"], matrix), ["Microsoft"])


if __name__ == "__main__":
    unittest.main()
