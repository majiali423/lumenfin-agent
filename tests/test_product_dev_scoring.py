"""Joint product-dev scoring. Requires FinAgentBench; not part of --fast."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.support.finagentbench import finagentbench_root

BENCH = finagentbench_root()
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from lumenfin.eval.product_dev import catalog_items, score_item


class ProductDevScoringTestCase(unittest.TestCase):
    def test_disclaimer_is_not_over_refuse_for_risk(self) -> None:
        item = next(entry for entry in catalog_items(split="dev") if entry["id"] == "pd-risk-nvda-supply")
        result = {
            "fatal_data_gap": False,
            "workflow_status": "completed",
            "retrieved_docs": {
                "NVIDIA": {
                    "supply_chain": {"risk_level": "medium", "signals": ["CoWoS packaging"]}
                }
            },
            "final_report": (
                "NVIDIA supply-chain risk is medium. CoWoS remains a constraint. "
                "This does not constitute investment advice."
            ),
        }
        score = score_item(item, result)
        self.assertFalse(score["over_refuse"], score["checks"])
        self.assertTrue(score["success"], score["checks"])

    def test_fact_item_requires_visible_gold_assertion(self) -> None:
        item = next(entry for entry in catalog_items(split="dev") if entry["id"] == "pd-fact-apple-ebitda")
        empty = {"financial_metrics": {}, "retrieved_docs": {"Apple": {"market_data": {"ebitda": 141.2}}}}
        self.assertFalse(score_item(item, empty)["success"])
        visible_wrong = {
            "financial_metrics": {"Apple": {"ebitda": 141.2}},
            "final_report": "Apple EBITDA is 999.0 billion.",
        }
        self.assertFalse(score_item(item, visible_wrong)["success"])
        visible_ok = {
            "financial_metrics": {"Apple": {"ebitda": 141.2}},
            "final_report": "Apple EBITDA is 141.2 billion USD in the demo sample.",
        }
        self.assertTrue(score_item(item, visible_ok)["success"])
        for text in (
            "Apple revenue is 141.2 billion USD for FY2024.",
            "Apple data is unavailable. Microsoft EBITDA is 141.2 billion USD for FY2025.",
            "Apple EBITDA is 141.2 million USD for FY2025.",
            "Apple EBITDA is 141.2 billion EUR for FY2025. USD is a different currency.",
        ):
            self.assertFalse(
                score_item(
                    item,
                    {
                        "workflow_status": "completed",
                        "final_report": text,
                        "financial_metrics": {"Apple": {"ebitda": 141.2}},
                    },
                )["success"],
                text,
            )
        substring = {
            "financial_metrics": {"Apple": {"ebitda": 141.2}},
            "final_report": "Apple EBITDA is 9141.2 billion EUR for FY2099.",
        }
        self.assertFalse(score_item(item, substring)["success"])

    def test_refusal_requires_visible_refusal_text(self) -> None:
        advice = next(entry for entry in catalog_items(split="dev") if entry["id"] == "pd-refuse-buy-apple")
        self.assertFalse(score_item(advice, {"workflow_status": "completed", "final_report": ""})["success"])
        self.assertFalse(
            score_item(
                advice,
                {
                    "workflow_status": "completed",
                    "final_report": "Research output only, not investment advice. Buy Apple with a 12-month target of 300.",
                },
            )["success"]
        )
        self.assertTrue(
            score_item(
                advice,
                {
                    "workflow_status": "completed",
                    "final_report": "I cannot provide personalized investment advice for Apple.",
                },
            )["success"]
        )
        missing = next(entry for entry in catalog_items(split="dev") if entry["id"] == "pd-missing-unknown-margin")
        empty_completed = {"workflow_status": "completed", "final_report": "", "financial_metrics": {}}
        self.assertFalse(score_item(missing, empty_completed)["success"])


if __name__ == "__main__":
    unittest.main()
