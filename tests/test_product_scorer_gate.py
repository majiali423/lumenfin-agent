from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import product_scorer_gate as gate  # noqa: E402


class ProductScorerGateTestCase(unittest.TestCase):
    def test_empty_ref_is_config_failure_not_passed(self) -> None:
        payload = gate.evaluate(product_ref="")
        self.assertEqual(payload["status"], "empty_ref")
        self.assertFalse(payload["passed"])
        self.assertFalse(payload["config_ok"])
        self.assertEqual(payload["joint_tests"], "not_run")

    def test_invalid_and_frozen_refs_are_config_failures(self) -> None:
        for ref in ("main", "master", "HEAD", "v0.1.0-rc.3", "v0.1.0-rc.4"):
            payload = gate.evaluate(product_ref=ref)
            self.assertEqual(payload["status"], "invalid_ref", ref)
            self.assertFalse(payload["passed"], ref)
            self.assertEqual(payload["joint_tests"], "not_run", ref)

    def test_compatible_ref_without_tests_is_not_a_passed_gate(self) -> None:
        payload = gate.evaluate(product_ref="abc123def456")
        self.assertEqual(payload["status"], "compatible_ref")
        self.assertTrue(payload["config_ok"])
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["joint_tests"], "not_run")

    def test_missing_scorer_module_is_not_passed(self) -> None:
        with mock.patch.object(gate, "v3_scorer_present", return_value=False):
            payload = gate.evaluate(
                product_ref="abc123def456",
                check_module=True,
            )
        self.assertEqual(payload["status"], "missing_scorer")
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["joint_tests"], "not_run")

    def test_joint_test_failure_is_not_passed(self) -> None:
        with mock.patch.object(gate, "v3_scorer_present", return_value=True):
            payload = gate.evaluate(
                product_ref="abc123def456",
                check_module=True,
                run_joint=True,
                joint_runner=lambda: 1,
            )
        self.assertEqual(payload["status"], "tests_failed")
        self.assertFalse(payload["passed"])
        self.assertEqual(payload["joint_tests"], "failed")

    def test_compatible_ref_with_v3_and_joint_pass(self) -> None:
        with mock.patch.object(gate, "v3_scorer_present", return_value=True):
            payload = gate.evaluate(
                product_ref="abc123def456",
                check_module=True,
                run_joint=True,
                joint_runner=lambda: 0,
            )
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["joint_tests"], "passed")

    def test_workflow_does_not_downgrade_offline_to_fast(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("run_tests.py --skip-joint", ci)
        self.assertIn("Product quality v3", ci)
        self.assertIn("run_tests.py --joint-only", ci)
        self.assertIn("product_scorer_gate.py --require-ref", ci)
        self.assertNotIn("Offline tests without unpublished scorer", ci)
        offline_job = ci.split("name: Offline regression", 1)[1].split("product-quality:", 1)[0]
        self.assertIn("run_tests.py --skip-joint", offline_job)
        self.assertNotIn("run_tests.py --fast", offline_job)


if __name__ == "__main__":
    unittest.main()
