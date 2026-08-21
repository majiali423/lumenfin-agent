from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_VERSION = "0.1.0rc4"
PUBLISHED_TAG = "v0.1.0-rc.4"
CANDIDATE_VERSION = "0.1.0rc5"
CANDIDATE_TAG = "v0.1.0-rc.5"


class VersionConsistencyTests(unittest.TestCase):
    def test_current_candidate_version_is_consistent_across_runtime_surfaces(self) -> None:
        package_version = tomllib.loads(
            (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]["version"]
        app_source = (ROOT / "src" / "lumenfin" / "api" / "app.py").read_text(
            encoding="utf-8"
        )
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        repro = (ROOT / "docs" / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme_zh = (ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
        changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

        self.assertEqual(package_version, CANDIDATE_VERSION)
        self.assertIn(f'return "{package_version}"', app_source)
        image_defaults = re.findall(r"LUMENFIN_IMAGE_TAG:-(0\.1\.0rc\d+)", compose)
        self.assertEqual(image_defaults, [package_version] * 4)
        self.assertIn(f"lumenfin-agent:{package_version}", repro)
        self.assertNotIn(f"lumenfin-agent:{PUBLISHED_VERSION}", repro)

        self.assertIn(PUBLISHED_VERSION, readme)
        self.assertIn(PUBLISHED_TAG, readme)
        self.assertIn(CANDIDATE_VERSION, readme)
        self.assertIn(CANDIDATE_TAG, readme)
        self.assertRegex(readme, r"not tagged and\s+not released")
        self.assertNotIn("current-main regression", readme)
        self.assertIn("2026-08-13 post-rc4 snapshot", readme)

        self.assertIn(PUBLISHED_VERSION, readme_zh)
        self.assertIn(CANDIDATE_VERSION, readme_zh)
        self.assertIn(CANDIDATE_TAG, readme_zh)
        self.assertRegex(readme_zh, r"尚未打标签、尚未发布")
        self.assertIn("2026-08-13 post-rc4 快照", readme_zh)

        self.assertIn("candidate preparation (unpublished)", changelog)
        self.assertNotIn("## Unreleased", changelog.split("## 0.1.0rc4", 1)[0])
        self.assertIn(f"## {PUBLISHED_VERSION} — 2026-08-13", changelog)
        self.assertNotRegex(
            changelog.split("## 0.1.0rc4", 1)[0],
            r"https://github.com/majiali423/lumenfin-agent/releases/tag/v0\.1\.0-rc\.5",
        )


if __name__ == "__main__":
    unittest.main()
