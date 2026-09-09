"""Portable discovery of the sibling FinAgentBench repository."""

from __future__ import annotations

import os
from pathlib import Path


def lumenfin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def finagentbench_root() -> Path:
    configured = os.getenv("FINAGENTBENCH_DIR", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if (candidate / "finagentbench").is_dir():
            return candidate.resolve()
        raise RuntimeError(f"FINAGENTBENCH_DIR={candidate} has no finagentbench package.")
    allow_sibling = os.getenv("LUMENFIN_ALLOW_SIBLING_FAB", "").strip() in {"1", "true", "yes"}
    if allow_sibling:
        sibling = lumenfin_root().parent / "finagentbench-demo"
        if (sibling / "finagentbench").is_dir():
            return sibling.resolve()
    raise RuntimeError(
        "FinAgentBench repository not found. Set FINAGENTBENCH_DIR to the "
        "finagentbench-demo checkout (unpublished product scorer: use the working "
        "tree; do not invent a SHA/tag). Sibling discovery requires "
        "LUMENFIN_ALLOW_SIBLING_FAB=1."
    )
