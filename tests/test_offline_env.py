from __future__ import annotations

import unittest

from scripts.offline_env import CREDENTIAL_KEYS, apply_offline_env, credential_source_report


class OfflineEnvTests(unittest.TestCase):
    def test_apply_offline_env_clears_existing_credential_process_values(self) -> None:
        environ = {
            "DEEPSEEK_API_KEY": "sk-should-never-be-used",
            "APP_ENV": "production",
            "DATA_MODE": "live",
        }
        applied = apply_offline_env(environ=environ)
        self.assertEqual(environ["DEEPSEEK_API_KEY"], "")
        self.assertEqual(environ["APP_ENV"], "test")
        self.assertEqual(environ["DATA_MODE"], "demo")
        self.assertEqual(applied["DEEPSEEK_API_KEY"], "<empty>")
        self.assertNotIn("sk-should-never-be-used", str(applied))

    def test_credential_report_never_includes_secret_values(self) -> None:
        report = credential_source_report()
        serialized = str(report)
        for item in report:
            self.assertIn(item["key"], CREDENTIAL_KEYS)
            self.assertNotIn("sk-", serialized.lower())
            self.assertTrue({"process_nonempty", "dotenv_length", "effective_source"} <= set(item))


if __name__ == "__main__":
    unittest.main()
