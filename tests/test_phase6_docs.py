"""Release boundaries: evaluator isolation, UTF-8, packaging and CI routing.

README wording and heading order are editorial choices. The documentation link
checker validates the published navigation without freezing presentation copy.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_utf8_subprocess(
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a child Python with UTF-8 stdio so GBK consoles cannot crash decoding."""
    merged = os.environ.copy() if env is None else dict(env)
    merged["PYTHONIOENCODING"] = "utf-8"
    merged["PYTHONUTF8"] = "1"
    if extra_env:
        merged.update(extra_env)
    completed = subprocess.run(
        args,
        cwd=str(cwd or ROOT),
        env=merged,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.stdout is None:
        completed.stdout = ""
    if completed.stderr is None:
        completed.stderr = ""
    return completed


class Phase6DocsTestCase(unittest.TestCase):
    def test_run_tests_exposes_fast_flag(self) -> None:
        src = (ROOT / "scripts" / "run_tests.py").read_text(encoding="utf-8")
        self.assertIn('"--fast"', src)
        self.assertIn('"--skip-joint"', src)
        self.assertIn("tests.test_phase6_docs", src)
        self.assertIn("tests.test_phase4_task_spec", src)
        from scripts.run_tests import FAST_MODULES, JOINT_MODULES

        self.assertNotIn("tests.test_product_dev_scoring", FAST_MODULES)
        self.assertIn("tests.test_product_dev_scoring", JOINT_MODULES)

        for name in FAST_MODULES:
            rel = Path(*name.split("."))
            path = ROOT / f"{rel}.py"
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertNotIn("finagentbench", imported, name)

    def test_fast_modules_run_when_finagentbench_import_is_blocked(self) -> None:
        probe = (
            "import os, sys, unittest\n"
            "class _Block:\n"
            "    def find_spec(self, name, path, target=None):\n"
            "        if name == 'finagentbench' or name.startswith('finagentbench.'):\n"
            "            raise ImportError('finagentbench blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            "from scripts.run_tests import FAST_MODULES\n"
            "loader = unittest.TestLoader()\n"
            "suite = unittest.TestSuite()\n"
            "for name in FAST_MODULES:\n"
            "    if name == 'tests.test_phase6_docs':\n"
            "        continue\n"
            "    suite.addTests(loader.loadTestsFromName(name))\n"
            "result = unittest.TextTestRunner(verbosity=0).run(suite)\n"
            "raise SystemExit(0 if result.wasSuccessful() else 1)\n"
        )
        env = os.environ.copy()
        env.pop("FINAGENTBENCH_DIR", None)
        completed = run_utf8_subprocess([sys.executable, "-c", probe], env=env)
        combined = (completed.stdout or "") + (completed.stderr or "")
        self.assertEqual(completed.returncode, 0, combined)

    def test_utf8_subprocess_keeps_chinese_output_and_nonzero_status(self) -> None:
        probe = (
            "import sys\n"
            "sys.stdout.write('中文stdout-验收通过\\n')\n"
            "sys.stderr.write('中文stderr-保留返回码\\n')\n"
            "raise SystemExit(7)\n"
        )
        completed = run_utf8_subprocess([sys.executable, "-c", probe])
        self.assertEqual(completed.returncode, 7)
        self.assertIn("中文stdout-验收通过", completed.stdout)
        self.assertIn("中文stderr-保留返回码", completed.stderr)

    def test_wheel_does_not_package_static(self) -> None:
        data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        find = data["tool"]["setuptools"]["packages"]["find"]
        self.assertEqual(find["where"], ["src"])
        self.assertEqual(data["tool"]["setuptools"]["package-dir"], {"": "src"})
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("COPY static", dockerfile)

    def test_ci_fast_job_gates_offline_and_contract(self) -> None:
        ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        self.assertIn("run_tests.py --fast", ci)
        self.assertIn("run_tests.py --skip-joint", ci)
        self.assertIn("run_tests.py --joint-only", ci)
        self.assertIn("needs: [fast]", ci)
        self.assertGreaterEqual(ci.count("needs: [fast]"), 3)
        self.assertIn("FINAGENTBENCH_PRODUCT_REF", ci)
        self.assertIn("Product quality v3", ci)
        self.assertNotIn("deadbeef", ci)


if __name__ == "__main__":
    unittest.main()
