#!/usr/bin/env python3
"""FinanceBench retrieval evaluation. Offline by default; remote needs --allow-remote."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.financebench.runner import main
from lumenfin.stdio import configure_stdio_utf8


if __name__ == "__main__":
    configure_stdio_utf8()
    bootstrap_dotenv(strict_conflicts=True)
    raise SystemExit(main())
