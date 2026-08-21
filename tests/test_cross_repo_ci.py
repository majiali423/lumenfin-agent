"""Regression tests for LumenFin cross-repository CI orchestration."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (str(SCRIPTS), str(ROOT), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import run_cross_repo_ci as cross  # noqa: E402


class CrossRepoCiTestCase(unittest.TestCase):
    def test_missing_finrun_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.json"
            with self.assertRaises(cross.CrossRepoCiError) as ctx:
                cross.require_finrun_file(path)
            self.assertIn("missing", str(ctx.exception).lower())
            self.assertIn(str(path), str(ctx.exception))

    def test_empty_finrun_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.json"
            path.write_text("   \n", encoding="utf-8")
            with self.assertRaises(cross.CrossRepoCiError) as ctx:
                cross.require_finrun_file(path)
            self.assertIn("empty", str(ctx.exception).lower())

    def test_invalid_schema_finrun_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            path.write_text(json.dumps({"entities": []}), encoding="utf-8")
            with self.assertRaises(cross.CrossRepoCiError) as ctx:
                cross.require_finrun_file(path)
            self.assertIn("schema_version", str(ctx.exception))

    def test_missing_mutation_report_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mutation.json"
            with self.assertRaises(cross.CrossRepoCiError) as ctx:
                cross.require_mutation_report(path)
            self.assertIn("missing", str(ctx.exception).lower())

    def test_empty_mutation_results_fail_even_if_file_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mutation.json"
            path.write_text(json.dumps({"passed": True, "items": []}), encoding="utf-8")
            with self.assertRaises(cross.CrossRepoCiError) as ctx:
                cross.require_mutation_report(path)
            self.assertIn("items", str(ctx.exception).lower())

    def test_paths_resolve_from_different_cwd(self) -> None:
        fab = Path(os.environ.get("FINAGENTBENCH_DIR") or ROOT.parent / "finagentbench-demo")
        if not (fab / "fixtures" / "lumenfin_state_sample.json").is_file():
            self.skipTest("sibling FinAgentBench fixtures unavailable")
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "nested" / "out"
            finrun_path = (out / "sample_finrun.json").resolve()
            other = Path(tmp) / "other_cwd"
            other.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(other)
                payload = cross.generate_sample_finrun(
                    lumen=ROOT.resolve(),
                    fab=fab.resolve(),
                    finrun_path=finrun_path,
                )
            finally:
                os.chdir(previous)
            self.assertTrue(finrun_path.is_file())
            self.assertTrue(payload.get("schema_version"))
            self.assertGreater(finrun_path.stat().st_size, 20)
            # Absolute out path must not depend on caller cwd.
            self.assertTrue(str(finrun_path).endswith(str(Path("nested") / "out" / "sample_finrun.json")) or finrun_path.name == "sample_finrun.json")

    def test_dependency_checkout_noise_not_treated_as_unexpected_dirty(self) -> None:
        lines = [
            "?? finagentbench-demo/",
            "?? finagentbench-demo/README.md",
            " M src/lumenfin/config.py",
        ]
        with mock.patch.object(cross, "_git_porcelain", return_value=lines):
            dirty = cross.lumenfin_unexpected_dirty(ROOT)
        self.assertEqual(dirty, [" M src/lumenfin/config.py"])

    def test_real_tracked_modification_is_detected(self) -> None:
        lines = [" M README.md"]
        with mock.patch.object(cross, "_git_porcelain", return_value=lines):
            dirty = cross.lumenfin_unexpected_dirty(ROOT)
        self.assertEqual(dirty, [" M README.md"])

    def test_required_ci_validates_frozen_pin_and_latest_published_release(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("lane: authoritative-frozen", ci)
        self.assertIn("finagentbench_ref: v0.1.0-rc.3", ci)
        self.assertIn("lane: latest-published", ci)
        self.assertIn("finagentbench_ref: v0.1.0-rc.4", ci)
        self.assertIn("finrun-contract-gate-${{ matrix.lane }}-${{ matrix.finagentbench_ref }}", ci)
        self.assertIn("fail-fast: false", ci)
        self.assertNotRegex(ci, r"(?m)^\s+ref:\s+master\s*$")
        self.assertNotIn("ref: ${{ env.FINAGENTBENCH_REF || 'master' }}", ci)

    def test_summary_records_requested_ref_and_lane(self) -> None:
        self.assertIn("finagentbench_requested_ref", Path(cross.__file__).read_text(encoding="utf-8"))
        self.assertIn("finagentbench_lane", Path(cross.__file__).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
