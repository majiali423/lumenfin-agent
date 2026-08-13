from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class VersionConsistencyTests(unittest.TestCase):
    def test_current_candidate_version_is_consistent_across_runtime_surfaces(self) -> None:
        package_version = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        app_source = (ROOT / "src" / "lumenfin" / "api" / "app.py").read_text(
            encoding="utf-8"
        )
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertEqual(package_version, "0.1.0rc4")
        self.assertIn(f'return "{package_version}"', app_source)
        image_defaults = re.findall(r"LUMENFIN_IMAGE_TAG:-(0\.1\.0rc\d+)", compose)
        self.assertEqual(image_defaults, [package_version] * 4)


if __name__ == "__main__":
    unittest.main()
