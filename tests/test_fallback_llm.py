from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.tools import extract_companies_from_query


class FallbackLLMTestCase(unittest.TestCase):
    def test_executive_summary_uses_query_company_not_apple_microsoft(self) -> None:
        client = LocalFallbackLLMClient()
        summary = client.chat(
            system_prompt="Write a 4-5 sentence executive summary in Chinese.",
            user_prompt="Query: NVIDIA FY2025 data center GPU revenue and gross margin trends.",
        )
        self.assertIn("NVIDIA", summary)
        self.assertNotIn("Apple", summary)
        self.assertNotIn("Microsoft", summary)

    def test_company_extractor_returns_nvidia(self) -> None:
        client = LocalFallbackLLMClient()
        raw = client.chat(
            system_prompt='返回 JSON 格式: {"companies": ["公司1"]}',
            user_prompt="NVIDIA FY2025 数据中心 GPU 收入",
        )
        companies = extract_companies_from_query("NVIDIA FY2025 数据中心 GPU 收入", llm_client=client)
        self.assertEqual(companies, ["NVIDIA"])


if __name__ == "__main__":
    unittest.main()
