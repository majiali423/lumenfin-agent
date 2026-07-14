from __future__ import annotations

import unittest

from lumenfin.tools import canonicalize_companies, has_computable_fundamentals


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


if __name__ == "__main__":
    unittest.main()
