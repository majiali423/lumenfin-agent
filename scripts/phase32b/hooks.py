from __future__ import annotations

import shutil
import time
from pathlib import Path

from .settings import OUTPUT_DIR

HOOK_ROOT = OUTPUT_DIR / "hooks"
CHECKPOINT_BARRIER = HOOK_ROOT / "checkpoint-barrier"
INDEX_PAUSE = HOOK_ROOT / "index-pause"


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def arm(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "release").unlink(missing_ok=True)
    (path / "armed").write_text("1\n", encoding="utf-8")


def release(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "release").write_text("1\n", encoding="utf-8")


def disarm(path: Path) -> None:
    (path / "armed").unlink(missing_ok=True)
    (path / "release").unlink(missing_ok=True)


def wait_for_ready_files(path: Path, *, count: int, timeout_seconds: float = 30.0) -> list[Path]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        ready = sorted(path.glob("ready-*"))
        if len(ready) >= count:
            return ready
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for {count} ready files in {path}")


def wait_for_claimed(path: Path, *, timeout_seconds: float = 60.0) -> Path:
    import json

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        claimed = sorted(path.glob("claimed-*.json"))
        for candidate in claimed:
            try:
                text = candidate.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                json.loads(text)
                return candidate
            except (OSError, json.JSONDecodeError):
                continue
        time.sleep(0.05)
    raise TimeoutError(f"Timed out waiting for claim marker in {path}")
