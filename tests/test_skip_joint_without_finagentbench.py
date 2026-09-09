"""--skip-joint must collect without FinAgentBench; joint modules still fail closed."""

from __future__ import annotations

import os
import sys
import unittest

from tests.test_phase6_docs import run_utf8_subprocess

_BLOCK = (
    "import os, sys, unittest\n"
    "os.environ.pop('FINAGENTBENCH_DIR', None)\n"
    "os.environ.pop('LUMENFIN_ALLOW_SIBLING_FAB', None)\n"
    "class _Block:\n"
    "    def find_spec(self, name, path, target=None):\n"
    "        if name == 'finagentbench' or name.startswith('finagentbench.'):\n"
    "            raise ImportError('finagentbench blocked')\n"
    "        return None\n"
    "sys.meta_path.insert(0, _Block())\n"
)


class SkipJointWithoutFinAgentBenchTests(unittest.TestCase):
    def test_skip_joint_collects_and_runs_rag_page_identity_without_scorer(self) -> None:
        probe = (
            _BLOCK
            + "from scripts.run_tests import _load_offline_suite\n"
            "suite = _load_offline_suite(unittest.TestLoader(), skip_joint=True)\n"
            "ids = []\n"
            "def collect(item):\n"
            "    for child in item:\n"
            "        if isinstance(child, unittest.TestSuite):\n"
            "            collect(child)\n"
            "        else:\n"
            "            ids.append(child.id())\n"
            "collect(suite)\n"
            "joined = '\\n'.join(ids)\n"
            "assert 'tests.test_rag_page_identity' in joined, ids[:20]\n"
            "assert 'tests.test_upload_product_loop' not in joined\n"
            "import tests.test_rag_page_identity as rag\n"
            "result = unittest.TextTestRunner(verbosity=0).run(\n"
            "    unittest.defaultTestLoader.loadTestsFromModule(rag)\n"
            ")\n"
            "raise SystemExit(0 if result.wasSuccessful() else 1)\n"
        )
        env = os.environ.copy()
        env.pop("FINAGENTBENCH_DIR", None)
        env.pop("LUMENFIN_ALLOW_SIBLING_FAB", None)
        completed = run_utf8_subprocess([sys.executable, "-c", probe], env=env)
        combined = (completed.stdout or "") + (completed.stderr or "")
        self.assertEqual(completed.returncode, 0, combined)

    def test_joint_upload_module_fails_closed_without_evaluator(self) -> None:
        probe = (
            _BLOCK
            + "try:\n"
            "    import tests.test_upload_product_loop\n"
            "except Exception as exc:\n"
            "    text = type(exc).__name__ + ': ' + str(exc)\n"
            "    print(text)\n"
            "    ok = 'FinAgentBench' in text or 'finagentbench' in text.lower()\n"
            "    raise SystemExit(0 if ok else 2)\n"
            "raise SystemExit(3)\n"
        )
        env = os.environ.copy()
        env.pop("FINAGENTBENCH_DIR", None)
        env.pop("LUMENFIN_ALLOW_SIBLING_FAB", None)
        completed = run_utf8_subprocess([sys.executable, "-c", probe], env=env)
        combined = (completed.stdout or "") + (completed.stderr or "")
        self.assertEqual(completed.returncode, 0, combined)
        self.assertNotIn("skipped", combined.lower())
