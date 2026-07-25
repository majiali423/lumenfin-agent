"""Regression: process env must not silently diverge from project .env."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import (  # noqa: E402
    EnvConflictError,
    assert_no_env_conflicts,
    bootstrap_dotenv,
    describe_credential_sources,
    detect_env_conflicts,
)


class EnvBootstrapConflictTestCase(unittest.TestCase):
    def test_process_env_shadowing_dotenv_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=sk-from-dotenv-file-value\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-process-shadow"}, clear=False):
                conflicts = detect_env_conflicts(root=root)
                self.assertTrue(any("DEEPSEEK_API_KEY" in item for item in conflicts))
                with self.assertRaises(EnvConflictError) as ctx:
                    assert_no_env_conflicts(root=root)
                message = str(ctx.exception)
                self.assertIn("process env shadows .env", message)
                self.assertIn("Unset the process variable", message)
                self.assertNotIn("sk-from-dotenv-file-value", message)
                self.assertNotIn("sk-process-shadow", message)
                with self.assertRaises(EnvConflictError):
                    bootstrap_dotenv(root=root, announce=False, strict_conflicts=True)

    def test_matching_process_and_dotenv_values_are_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=sk-same-value-everywhere\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-same-value-everywhere"}, clear=False):
                self.assertEqual(detect_env_conflicts(root=root), [])
                bootstrap_dotenv(root=root, announce=False, strict_conflicts=True)
                reports = describe_credential_sources(root=root, keys=("DEEPSEEK_API_KEY",))
                self.assertEqual(reports[0].source, "process_env")
                self.assertEqual(reports[0].length, len("sk-same-value-everywhere"))

    def test_empty_process_key_does_not_conflict_with_dotenv(self) -> None:
        """Empty process values are not treated as credential conflicts."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=sk-from-dotenv-file-value\n", encoding="utf-8")
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}, clear=False):
                self.assertEqual(detect_env_conflicts(root=root), [])

    def test_preflight_and_runtime_share_fail_fast_path(self) -> None:
        """Simulates the RC bug: check_deepseek-style override vs AppConfig path."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("DEEPSEEK_API_KEY=sk-dotenv-good-key-value\n", encoding="utf-8")
            polluted = {"DEEPSEEK_API_KEY": "sk-bad-short"}
            with patch.dict(os.environ, polluted, clear=False):
                # Formal path refuses to continue.
                with self.assertRaises(EnvConflictError):
                    bootstrap_dotenv(root=root, strict_conflicts=True)
                # Override would hide the pollution (forbidden for production path).
                from dotenv import load_dotenv

                load_dotenv(root / ".env", override=True)
                self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), "sk-dotenv-good-key-value")


if __name__ == "__main__":
    unittest.main()
