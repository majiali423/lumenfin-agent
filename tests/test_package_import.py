"""Regression tests for side-effect-free package imports."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


class PackageImportTestCase(unittest.TestCase):
    def test_import_does_not_create_app_connect_or_write_files(self) -> None:
        script = textwrap.dedent(
            """
            import importlib
            import socket
            import sqlite3
            from pathlib import Path
            from unittest.mock import patch

            before = set(Path.cwd().iterdir())
            with (
                patch.object(socket.socket, "connect", side_effect=AssertionError("network connection during import")),
                patch.object(sqlite3, "connect", side_effect=AssertionError("database connection during import")),
            ):
                package = importlib.import_module("lumenfin")
                api_module = importlib.import_module("lumenfin.api.app")

            after = set(Path.cwd().iterdir())
            assert callable(package.create_app)
            assert not hasattr(api_module, "app")
            assert after == before, (before, after)
            """
        )
        env = os.environ.copy()
        for name in (
            "APP_ENV",
            "MAS_API_KEY",
            "MAS_DATABASE_URL",
            "MAS_DB_PATH",
            "MAS_MILVUS_URI",
            "MAS_REDIS_URL",
            "POSTGRES_PASSWORD",
        ):
            env.pop(name, None)
        env["PYTHONPATH"] = str(SRC)
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=tmp,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
