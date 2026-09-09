from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def finagentbench_root() -> Path:
    """Locate FinAgentBench. Missing checkout is an error, not a silent skip.

    Prefer ``FINAGENTBENCH_DIR`` or an installed package with fixtures.
    Sibling-directory discovery is last-resort only so isolated venvs can
    refuse an accidental neighbor checkout.
    """
    env = os.environ.get("FINAGENTBENCH_DIR", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        if (path / "finagentbench").is_dir():
            return path
        raise RuntimeError(
            f"FINAGENTBENCH_DIR={path} does not contain the finagentbench package."
        )
    spec = importlib.util.find_spec("finagentbench")
    if spec and spec.origin:
        root = Path(spec.origin).resolve().parent.parent
        if (root / "finagentbench").is_dir() and (
            (root / "fixtures").is_dir() or (root / "pyproject.toml").is_file()
        ):
            return root
    if os.environ.get("LUMENFIN_ALLOW_SIBLING_FAB", "").strip() in {"1", "true", "yes"}:
        repo = Path(__file__).resolve().parents[2]
        for path in (repo.parent / "finagentbench-demo", repo / "finagentbench-demo"):
            if (path / "finagentbench").is_dir():
                return path
    raise RuntimeError(
        "FinAgentBench is required for this product-quality test. "
        "pip install -e <finagentbench-demo> and set FINAGENTBENCH_DIR to that "
        "checkout. Sibling auto-discovery is off unless LUMENFIN_ALLOW_SIBLING_FAB=1. "
        "A missing evaluator must not be reported as a passed joint gate. "
        "Do not invent a SHA/tag for an unpublished scorer."
    )
