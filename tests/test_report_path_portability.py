"""Guard published documentation against machine-local absolute paths."""
from __future__ import annotations

import re
import unittest
from scripts.check_doc_links import ROOT, TARGETS

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
    def test_release_docs_have_no_machine_absolute_paths(self) -> None:
        # Share the maintained release-document list with the link checker.
        # Historical reports/ directories need not exist in a fresh checkout.
        offenders: list[str] = []
        self.assertTrue(TARGETS, "release documentation must not be empty")
        for rel in TARGETS:
            path = ROOT / rel
            self.assertTrue(path.is_file(), f"missing release document: {rel}")
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                if _MACHINE_PATH.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()[:120]}")
        self.assertEqual(
            offenders,
            [],
            "release docs must use repo-relative paths, not machine absolute paths:\n"
            + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
