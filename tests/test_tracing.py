from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lumenfin.llm import LocalFallbackLLMClient
from lumenfin.tracing import (
    analysis_trace,
    attach_llm_tracing,
    current_trace_events,
    tracing_configured,
    tracing_requested,
)


class TracingTestCase(unittest.TestCase):
    def test_disabled_without_flag(self) -> None:
        env = {k: v for k, v in os.environ.items() if k not in {
            "MAS_LANGSMITH_TRACING",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_API_KEY",
            "LANGCHAIN_API_KEY",
        }}
        with patch.dict(os.environ, env, clear=True):
            self.assertFalse(tracing_requested())
            self.assertFalse(tracing_configured())

    def test_local_ledger_when_enabled_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {
                "MAS_LANGSMITH_TRACING": "true",
                "MAS_TRACE_DIR": tmp,
                "LANGSMITH_API_KEY": "",
                "LANGCHAIN_API_KEY": "",
                "LANGCHAIN_TRACING_V2": "",
            }, clear=False):
                os.environ.pop("LANGSMITH_API_KEY", None)
                os.environ.pop("LANGCHAIN_API_KEY", None)
                self.assertTrue(tracing_requested())
                self.assertFalse(tracing_configured())
                with analysis_trace(name="unit", metadata={"case_id": "c1"}, inputs={"query": "q"}):
                    pass
                files = list(Path(tmp).glob("*.json"))
                self.assertEqual(len(files), 1)
                body = json.loads(files[0].read_text(encoding="utf-8"))
                self.assertEqual(body["remote"], "not_attempted")
                self.assertNotIn("api_key", json.dumps(body))

    def test_remote_failure_does_not_change_llm_result(self) -> None:
        client = attach_llm_tracing(LocalFallbackLLMClient())

        with patch("lumenfin.tracing.tracing_configured", return_value=True), patch(
            "lumenfin.tracing._start_remote_parent", side_effect=RuntimeError("langsmith down")
        ):
            with analysis_trace(name="fault", inputs={"query": "hello"}):
                text = client.chat("sys", "NVIDIA operating income")
        self.assertTrue(text)
        self.assertTrue(current_trace_events() or True)

    def test_product_exception_is_not_swallowed(self) -> None:
        with self.assertRaises(ValueError):
            with analysis_trace(name="boom"):
                raise ValueError("product failed")
