"""Portable discovery of the sibling FinAgentBench repository."""

from __future__ import annotations

import os
from pathlib import Path


def lumenfin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def finagentbench_root() -> Path:
    configured = os.getenv("FINAGENTBENCH_DIR")
    candidates = [
        Path(configured).expanduser() if configured else None,
        lumenfin_root().parent / "finagentbench-demo",
    ]
    for candidate in candidates:
        if candidate and (candidate / "finagentbench").is_dir():
            return candidate.resolve()
    raise RuntimeError(
        "FinAgentBench repository not found. Set FINAGENTBENCH_DIR or clone "
        "finagentbench-demo next to lumenfin-agent."
    )
