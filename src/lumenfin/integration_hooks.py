"""Env-gated hooks for multi-process integration tests.

All hooks are no-ops unless:
1. the corresponding MAS_INTEGRATION_* directory env var is set, and
2. an ``armed`` file exists in that directory.

They must never sleep or block in normal deployments.
"""

from __future__ import annotations

import os
import time
from pathlib import Path


def _env_path(name: str) -> Path | None:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    return Path(raw)


def _armed(directory: Path) -> bool:
    return (directory / "armed").is_file()


def maybe_barrier_after_checkpoint_read(thread_id: str, expected_revision: int) -> None:
    """Coordinate cross-process CAS races after revision is read."""
    barrier_dir = _env_path("MAS_INTEGRATION_CHECKPOINT_BARRIER_DIR")
    if barrier_dir is None or not _armed(barrier_dir):
        return
    barrier_dir.mkdir(parents=True, exist_ok=True)
    worker = (os.getenv("MAS_WORKER_ID") or f"pid-{os.getpid()}").strip()
    ready = barrier_dir / f"ready-{worker}"
    ready.write_text(
        f"thread_id={thread_id}\nrevision={expected_revision}\npid={os.getpid()}\n",
        encoding="utf-8",
    )
    release = barrier_dir / "release"
    deadline = time.monotonic() + float(os.getenv("MAS_INTEGRATION_BARRIER_TIMEOUT_SECONDS", "30"))
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Integration checkpoint barrier timed out waiting for {release}"
            )
        time.sleep(0.05)


def maybe_pause_after_index_claim(
    *,
    document_id: str,
    tenant_id: str,
    index_owner: str,
    index_attempt: int,
) -> None:
    """Pause a worker after a successful lease claim until release file appears."""
    pause_dir = _env_path("MAS_INTEGRATION_INDEX_PAUSE_DIR")
    if pause_dir is None or not _armed(pause_dir):
        return
    pause_dir.mkdir(parents=True, exist_ok=True)
    worker = (os.getenv("MAS_WORKER_ID") or f"pid-{os.getpid()}").strip()
    claimed = pause_dir / f"claimed-{worker}.json"
    claimed.write_text(
        (
            "{\n"
            f'  "document_id": "{document_id}",\n'
            f'  "tenant_id": "{tenant_id}",\n'
            f'  "index_owner": "{index_owner}",\n'
            f'  "index_attempt": {index_attempt},\n'
            f'  "pid": {os.getpid()},\n'
            f'  "worker_id": "{worker}"\n'
            "}\n"
        ),
        encoding="utf-8",
    )
    release = pause_dir / "release"
    deadline = time.monotonic() + float(os.getenv("MAS_INTEGRATION_PAUSE_TIMEOUT_SECONDS", "120"))
    while not release.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Integration index pause timed out waiting for {release}")
        time.sleep(0.05)
