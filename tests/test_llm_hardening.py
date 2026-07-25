from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.llm import (
    DeepSeekChatClient,
    LLMSettings,
    LocalFallbackLLMClient,
    ResilientLLMClient,
    _is_company_extract_prompt,
)


class LlmHardeningTestCase(unittest.TestCase):
    def test_settings_strips_api_key_whitespace(self) -> None:
        with patch.dict(
            "os.environ",
            {"DEEPSEEK_API_KEY": "  sk-testkey  ", "DEEPSEEK_BASE_URL": " https://api.deepseek.com "},
            clear=False,
        ):
            settings = LLMSettings.from_env()
        self.assertEqual(settings.api_key, "sk-testkey")
        self.assertEqual(settings.base_url, "https://api.deepseek.com")

    def test_planner_structure_prompt_is_not_company_extract(self) -> None:
        prompt = (
            'return only json with keys: {"companies":["..."],'
            '"time_range":{"raw":"...","has_time":true},'
            '"prefer_uploaded_only":false}'
        )
        self.assertFalse(_is_company_extract_prompt(prompt.lower()))

    def test_company_extract_prompt_detected(self) -> None:
        self.assertTrue(_is_company_extract_prompt("你是一个公司名称提取器。返回 json"))

    def test_resilient_records_fallback_reason(self) -> None:
        primary = MagicMock()
        primary.backend_name = "deepseek"
        primary.model_name = "deepseek-chat"
        primary._usage_totals = {"prompt_tokens": 0, "completion_tokens": 0}
        primary.chat.side_effect = httpx.HTTPStatusError(
            "unauthorized",
            request=MagicMock(),
            response=MagicMock(status_code=401),
        )
        client = ResilientLLMClient(primary=primary, fallback=LocalFallbackLLMClient())
        content = client.chat("system", "NVIDIA executive summary")
        self.assertIn("NVIDIA", content)
        self.assertTrue(client.used_fallback)
        self.assertIsNotNone(client.last_error)
        self.assertIn("HTTPStatusError", client.last_error or "")


if __name__ == "__main__":
    unittest.main()
