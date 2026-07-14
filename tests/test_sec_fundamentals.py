from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from lumenfin.sec_fundamentals import fetch_sec_companyfacts_fundamentals
from lumenfin.tools import retrieve_company_payload


class SecFundamentalsTests(unittest.TestCase):
    def test_fetch_sec_maps_revenue_op_income_rd(self) -> None:
        facts = {
            "entityName": "NVIDIA CORP",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-01-26",
                                    "val": 130497000000,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-02-26",
                                }
                            ]
                        }
                    },
                    "OperatingIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-01-26",
                                    "val": 81453000000,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-02-26",
                                }
                            ]
                        }
                    },
                    "ResearchAndDevelopmentExpense": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-01-26",
                                    "val": 12914000000,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-02-26",
                                }
                            ]
                        }
                    },
                    "DepreciationAndAmortization": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2025-01-26",
                                    "val": 1864000000,
                                    "fy": 2025,
                                    "fp": "FY",
                                    "form": "10-K",
                                    "filed": "2025-02-26",
                                }
                            ]
                        }
                    },
                }
            },
        }

        mock_client = MagicMock()
        facts_resp = MagicMock()
        facts_resp.status_code = 200
        facts_resp.json.return_value = facts
        facts_resp.raise_for_status = MagicMock()
        mock_client.get.return_value = facts_resp

        with patch("lumenfin.sec_fundamentals.resolve_cik", return_value="0001045810"):
            payload = fetch_sec_companyfacts_fundamentals("NVDA", client=mock_client)

        assert payload is not None
        self.assertEqual(payload["structured_source"], "sec_companyfacts")
        self.assertAlmostEqual(payload["market_data"]["revenue_2025"], 130.497, places=3)
        self.assertAlmostEqual(payload["market_data"]["operating_income_2025"], 81.453, places=3)
        self.assertAlmostEqual(payload["market_data"]["r_and_d_2025"], 12.914, places=3)
        self.assertIn("ebitda_2025", payload["market_data"])

    def test_retrieve_prefers_sec_over_yahoo(self) -> None:
        sec = {
            "market_data": {"revenue_2025": 1.0, "operating_income_2025": 0.4, "r_and_d_2025": 0.2},
            "structured_source": "sec_companyfacts",
            "supply_chain": {"risk_level": "unknown", "signals": []},
            "earnings_call_quotes": [],
            "fundamentals_meta": {"provider": "sec_edgar", "symbol": "AAPL"},
        }
        yahoo = {
            "market_data": {"revenue_2025": 9.0, "ebitda_2025": 4.0},
            "structured_source": "yahoo_fundamentals",
            "supply_chain": {"risk_level": "unknown", "signals": []},
            "earnings_call_quotes": [],
        }
        with (
            patch("lumenfin.sec_fundamentals.fetch_sec_companyfacts_fundamentals", return_value=sec),
            patch("lumenfin.fundamentals.fetch_yahoo_fundamentals", return_value=yahoo),
        ):
            payload = retrieve_company_payload(
                "Apple",
                allow_sample_data=False,
                ticker="AAPL",
                fetch_live_fundamentals=True,
                fetch_sec_fundamentals=True,
            )
        self.assertEqual(payload["structured_source"], "sec_companyfacts")
        self.assertEqual(payload["market_data"]["revenue_2025"], 1.0)


if __name__ == "__main__":
    unittest.main()
