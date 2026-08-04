"""Pytest bootstrap: keep unit tests on SQLite-friendly APP_ENV=test."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.rag.profiles import apply_ci_rag_env

apply_ci_rag_env()
os.environ.setdefault("APP_ENV", "test")
