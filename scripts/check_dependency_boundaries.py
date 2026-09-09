#!/usr/bin/env python3
"""Record runtime vs optional extras and whether FinRun export needs PyMuPDF.

Does not rebuild the developer virtualenv. Isolated full LumenFin install
(Milvus) is optional and off by default.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def runtime_dependency_names(project: dict) -> list[str]:
    return list(project.get("project", {}).get("dependencies") or [])


def optional_groups(project: dict) -> dict[str, list[str]]:
    return dict(project.get("project", {}).get("optional-dependencies") or {})


def pip_check() -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip(),
    }


def export_without_fitz() -> dict[str, object]:
    probe = r"""
import importlib.abc
import sys
from pathlib import Path

src = Path(r""" + json.dumps(str(SRC)) + r""")
sys.path.insert(0, str(src))

class _BlockFitz(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "fitz" or fullname.startswith("fitz."):
            raise ModuleNotFoundError("fitz blocked for FinRun export probe")
        return None

sys.meta_path.insert(0, _BlockFitz())
from lumenfin.finrun import export_finrun_state
print("ok", callable(export_finrun_state))
"""
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "APP_ENV": "test"},
    )
    return {
        "returncode": proc.returncode,
        "stdout": (proc.stdout or "").strip(),
        "stderr": (proc.stderr or "").strip()[-2000:],
    }


def main() -> int:
    project = _pyproject()
    report = {
        "runtime_dependencies": runtime_dependency_names(project),
        "optional_dependency_groups": {name: list(pkgs) for name, pkgs in optional_groups(project).items()},
        "notes": {
            "pymupdf": "Runtime extra for PDF ingest (documents.parse_pdf_document). FinRun export must not import fitz.",
            "dev": "ruff/mypy for changed modules; not required to run the API.",
            "mcp-agent": "Optional LangChain MCP adapter; core ratios stay on SafeExpressionEvaluator.",
            "eval": "src/lumenfin/eval stays in-tree; this check does not relocate sealed holdout data.",
            "venv": "Does not upgrade or recreate the existing developer .venv.",
        },
        "pip_check": pip_check(),
        "finrun_export_without_fitz": export_without_fitz(),
    }
    print(json.dumps(report, indent=2))
    export_ok = report["finrun_export_without_fitz"]["returncode"] == 0
    return 0 if export_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
