from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FinRunExportImportTestCase(unittest.TestCase):
    def test_export_finrun_state_imports_without_fitz(self) -> None:
        probe = (
            "import importlib.abc, sys\n"
            f"sys.path.insert(0, r'{SRC}')\n"
            "class BlockFitz(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path, target=None):\n"
            "        if fullname == 'fitz' or fullname.startswith('fitz.'):\n"
            "            raise ModuleNotFoundError('fitz blocked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, BlockFitz())\n"
            "from lumenfin.finrun import export_finrun_state\n"
            "assert callable(export_finrun_state)\n"
            "print('export_ok')\n"
        )
        env = os.environ.copy()
        env["APP_ENV"] = "test"
        proc = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("export_ok", proc.stdout)


class QuantContractTestCase(unittest.TestCase):
    def test_has_computable_fundamentals_requires_revenue_and_numerator(self) -> None:
        from lumenfin.quant_contract import has_computable_fundamentals

        self.assertFalse(has_computable_fundamentals({"market_data": {"revenue": 10}}))
        self.assertTrue(
            has_computable_fundamentals({"market_data": {"revenue": 10, "ebitda": 3}})
        )

    def test_tools_reexports_quant_contract(self) -> None:
        from lumenfin import quant_contract, tools

        self.assertIs(tools.has_computable_fundamentals, quant_contract.has_computable_fundamentals)
        self.assertEqual(tools.AST_RATIO_KEYS, quant_contract.AST_RATIO_KEYS)


class ReportGapAndApiMappingTestCase(unittest.TestCase):
    def test_fatal_gap_demo_disclaimer(self) -> None:
        from lumenfin.agents.report_gap import render_fatal_data_gap_sections

        sections, detail = render_fatal_data_gap_sections(
            {"companies": ["Apple"], "data_gap_detail": "missing metrics"},
            data_mode="demo",
        )
        text = "\n".join(sections)
        self.assertEqual(detail, "missing metrics")
        self.assertIn("DEMO MODE", text)
        self.assertIn("Apple", text)

    def test_public_job_strips_execution_token(self) -> None:
        from lumenfin.api.responses import public_job

        public = public_job(
            {
                "job_id": "j1",
                "execution_token": "secret",
                "result": {"thread_id": "t1", "workflow_status": "completed"},
            },
            data_mode="demo",
        )
        self.assertNotIn("execution_token", public)
        self.assertEqual(public["result"]["data_mode"], "demo")

    def test_runtime_exposes_dependency_protocol(self) -> None:
        from lumenfin.agents.dependencies import RuntimeDependencies
        from lumenfin.agents.runtime import AgentRuntime
        from lumenfin.knowledge_store import InMemoryKnowledgeStore
        from lumenfin.llm import LocalFallbackLLMClient
        from lumenfin.memory import ReasoningMemory, SessionMemory

        class _FakeMarket:
            def fetch_company_snapshot(self, company: str, symbol: str | None = None) -> dict:
                return {}

        runtime = AgentRuntime(
            session_memory=SessionMemory(),
            knowledge_memory=InMemoryKnowledgeStore(),
            reasoning_memory=ReasoningMemory(),
            llm_client=LocalFallbackLLMClient(),
            market_data_client=_FakeMarket(),  # type: ignore[arg-type]
            rag_enabled=False,
        )
        deps = runtime.dependencies()
        self.assertIs(deps, runtime)
        self.assertIsInstance(deps, AgentRuntime)
        _: RuntimeDependencies = deps


if __name__ == "__main__":
    unittest.main()
