from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from lumenfin.sec_fundamentals import _user_agent, fetch_sec_companyfacts_fundamentals
from lumenfin.tools import retrieve_company_payload


class SecFundamentalsTests(unittest.TestCase):
    def test_production_requires_sec_operator_identity(self) -> None:
        with patch.dict(os.environ, {"APP_ENV": "production"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "SEC_USER_AGENT is required"):
                _user_agent()

    def test_configured_sec_operator_identity_is_used(self) -> None:
        with patch.dict(
            os.environ,
            {"APP_ENV": "production", "SEC_USER_AGENT": "LumenFin/0.1 ops@example.com"},
            clear=True,
        ):
            self.assertEqual(_user_agent(), "LumenFin/0.1 ops@example.com")

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
        self.assertAlmostEqual(payload["market_data"]["revenue"], 130.497, places=3)
        self.assertAlmostEqual(payload["market_data"]["operating_income"], 81.453, places=3)
        self.assertAlmostEqual(payload["market_data"]["r_and_d"], 12.914, places=3)
        self.assertIn("ebitda", payload["market_data"])

    def test_fetch_sec_retries_transient_facts_failure(self) -> None:
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
                }
            },
        }

        ok_resp = MagicMock()
        ok_resp.status_code = 200
        ok_resp.json.return_value = facts
        ok_resp.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.side_effect = [TimeoutError("temporary SEC timeout"), ok_resp]

        with (
            patch("lumenfin.sec_fundamentals.resolve_cik", return_value="0001045810"),
            patch("lumenfin.sec_fundamentals.time.sleep", return_value=None),
        ):
            payload = fetch_sec_companyfacts_fundamentals("NVDA", client=mock_client)

        assert payload is not None
        self.assertEqual(mock_client.get.call_count, 2)
        self.assertEqual(payload["structured_source"], "sec_companyfacts")
        self.assertAlmostEqual(payload["market_data"]["revenue"], 130.497, places=3)

    def test_retrieve_prefers_sec_over_yahoo(self) -> None:
        sec = {
            "market_data": {"revenue": 1.0, "operating_income": 0.4, "r_and_d": 0.2},
            "structured_source": "sec_companyfacts",
            "supply_chain": {"risk_level": "unknown", "signals": []},
            "earnings_call_quotes": [],
            "fundamentals_meta": {"provider": "sec_edgar", "symbol": "AAPL"},
        }
        yahoo = {
            "market_data": {"revenue": 9.0, "ebitda": 4.0},
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
        self.assertEqual(payload["market_data"]["revenue"], 1.0)


if __name__ == "__main__":
    unittest.main()
