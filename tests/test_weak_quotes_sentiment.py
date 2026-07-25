from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.tools import analyze_sentiment_deep, quotes_are_weak_for_llm


class _CountingLLM:
    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *args: Any, **kwargs: Any) -> str:
        self.calls += 1
        return (
            '{"overall_tone": "bullish", "confidence_score": 8, '
            '"key_themes": ["growth"], "risk_flags": [], "strategic_priority": "AI"}'
        )


class WeakQuotesSentimentTestCase(unittest.TestCase):
    def test_empty_and_short_quotes_are_weak(self) -> None:
        self.assertTrue(quotes_are_weak_for_llm([]))
        self.assertTrue(quotes_are_weak_for_llm(["short"]))
        self.assertTrue(quotes_are_weak_for_llm(["Profile generation pending for Apple."]))

    def test_substantive_quotes_are_not_weak(self) -> None:
        quote = (
            "We remain confident in durable growth as AI infrastructure demand accelerates, "
            "while managing supply-chain constraints carefully across key geographies."
        )
        self.assertFalse(quotes_are_weak_for_llm([quote]))

    def test_deep_skips_llm_for_weak_quotes(self) -> None:
        llm = _CountingLLM()
        result = analyze_sentiment_deep(["n/a"], llm_client=llm)
        self.assertEqual(llm.calls, 0)
        self.assertTrue(result.get("llm_skipped"))
        self.assertEqual(result.get("skip_reason"), "weak_quotes")

    def test_deep_calls_llm_for_substantive_quotes(self) -> None:
        llm = _CountingLLM()
        quote = (
            "We remain confident in durable growth as AI infrastructure demand accelerates, "
            "while managing supply-chain constraints carefully across key geographies."
        )
        result = analyze_sentiment_deep([quote], llm_client=llm)
        self.assertEqual(llm.calls, 1)
        self.assertFalse(result.get("llm_skipped"))
        self.assertEqual(result.get("label"), "bullish")


if __name__ == "__main__":
    unittest.main()
