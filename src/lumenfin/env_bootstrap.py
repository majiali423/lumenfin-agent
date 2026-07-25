"""Shared dotenv bootstrap for preflight and runtime.

Preserves normal environment-variable precedence (process env wins).
Never calls ``load_dotenv(override=True)``. When a non-empty process value
conflicts with a non-empty project ``.env`` value, fail fast so preflight and
formal runs cannot silently diverge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import dotenv_values, load_dotenv

# Credential-like keys that must not silently diverge between process env and .env.
_WATCHED_KEYS: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "ALPHAVANTAGE_API_KEY",
    "MAS_API_KEY",
)


class EnvConflictError(RuntimeError):
    """Raised when process env and project .env disagree on a credential value."""


@dataclass(frozen=True)
class CredentialSourceReport:
    key: str
    source: str
    length: int


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _length(value: str | None) -> int:
    return len((value or "").strip())


def describe_credential_sources(
    *,
    root: Path | None = None,
    keys: Iterable[str] = _WATCHED_KEYS,
) -> list[CredentialSourceReport]:
    root = root or project_root()
    file_vals = dotenv_values(root / ".env")
    cwd_vals = dotenv_values(Path.cwd() / ".env") if Path.cwd().resolve() != root.resolve() else {}
    reports: list[CredentialSourceReport] = []
    for key in keys:
        proc = os.environ.get(key)
        if proc is not None and proc.strip():
            source = "process_env"
            length = _length(proc)
        elif (file_vals.get(key) or "").strip():
            source = "project_dotenv"
            length = _length(file_vals.get(key))
        elif (cwd_vals.get(key) or "").strip():
            source = "cwd_dotenv"
            length = _length(cwd_vals.get(key))
        else:
            source = "unset"
            length = 0
        reports.append(CredentialSourceReport(key=key, source=source, length=length))
    return reports


def detect_env_conflicts(
    *,
    root: Path | None = None,
    keys: Iterable[str] = _WATCHED_KEYS,
) -> list[str]:
    """Return human-readable conflict messages (no secret values)."""
    root = root or project_root()
    file_vals = dotenv_values(root / ".env")
    conflicts: list[str] = []
    for key in keys:
        proc = os.environ.get(key)
        file_val = file_vals.get(key)
        proc_stripped = (proc or "").strip()
        file_stripped = (file_val or "").strip()
        if proc_stripped and file_stripped and proc_stripped != file_stripped:
            conflicts.append(
                f"{key}: process_env(len={len(proc_stripped)}) conflicts with "
                f"project_dotenv(len={len(file_stripped)}). Unset the process "
                f"variable or align it with {root / '.env'} before continuing."
            )
    return conflicts


def assert_no_env_conflicts(*, root: Path | None = None) -> None:
    conflicts = detect_env_conflicts(root=root)
    if conflicts:
        joined = "\n".join(f"  - {item}" for item in conflicts)
        raise EnvConflictError(
            "Environment credential conflict detected (process env shadows .env).\n"
            f"{joined}\n"
            "Refusing to continue so preflight and runtime cannot diverge. "
            "Do not use load_dotenv(override=True) to hide this."
        )


def announce_credential_sources(*, root: Path | None = None) -> None:
    for report in describe_credential_sources(root=root):
        print(
            f"credential {report.key}: source={report.source} len={report.length}",
            flush=True,
        )


def bootstrap_dotenv(
    *,
    root: Path | None = None,
    announce: bool = False,
    strict_conflicts: bool = True,
) -> Path:
    """Load dotenv files without override; optionally fail on conflicts."""
    root = root or project_root()
    if strict_conflicts:
        assert_no_env_conflicts(root=root)
    # Process env wins; .env only fills missing keys.
    load_dotenv(root / ".env")
    load_dotenv()
    if announce:
        announce_credential_sources(root=root)
    return root
