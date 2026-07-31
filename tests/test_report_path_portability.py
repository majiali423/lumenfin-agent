"""Guard tracked reports against machine-local absolute paths."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

# Windows drive paths (not URL schemes like https://), and home directories.
_MACHINE_PATH = re.compile(
    r"(?i)"
    r"("
    r"(?<![A-Za-z])[a-z]:(?:\\|/)|"
    r"/Users/[A-Za-z0-9_.-]+|"
    r"/home/[A-Za-z0-9_.-]+"
    r")"
)


class ReportPathPortabilityTestCase(unittest.TestCase):
    def test_reports_have_no_machine_absolute_paths(self) -> None:
        self.assertTrue(REPORTS.is_dir(), f"missing reports dir: {REPORTS}")
        offenders: list[str] = []
        for path in sorted(REPORTS.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _MACHINE_PATH.search(line):
                    rel = path.relative_to(ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")
        self.assertEqual(
            offenders,
            [],
            "reports must use repo-relative paths, not machine absolute paths:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
