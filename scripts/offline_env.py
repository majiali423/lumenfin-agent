"""Force a credential-free offline process environment.

Import this module before any ``lumenfin`` import. It never prints secret
values; credential reports include only source labels and lengths.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, MutableMapping

CREDENTIAL_KEYS: tuple[str, ...] = (
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "ALPHAVANTAGE_API_KEY",
    "MAS_API_KEY",
    "MAS_API_KEY_PRINCIPALS",
)

OFFLINE_BASE_ENV: dict[str, str] = {
    "APP_ENV": "test",
    "DATA_MODE": "demo",
    "ALLOW_LOCAL_FALLBACK": "true",
    "DEEPSEEK_API_KEY": "",
    "DASHSCOPE_API_KEY": "",
    "ALPHAVANTAGE_API_KEY": "",
    "MAS_API_KEY": "",
    "MAS_API_KEY_PRINCIPALS": "",
    "MAS_REDIS_URL": "",
    "MAS_FETCH_LIVE_FUNDAMENTALS": "false",
    "MAS_FETCH_SEC_FUNDAMENTALS": "false",
    "MAS_RAG_ENABLED": "true",
    "MAS_RAG_INDEX_MODE": "sync_on_run",
    "MAS_EMBEDDING_PROVIDER": "deterministic",
    "MAS_EMBEDDING_DIMENSION": "384",
    "DASHSCOPE_EMBEDDING_DIMENSION": "384",
    "MAS_RAG_RERANK_ENABLED": "true",
    "MAS_RAG_RERANK_PROVIDER": "lexical",
}


def apply_offline_env(
    extra: Mapping[str, str] | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Overwrite process env with offline-safe values. Returns applied keys only."""
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    applied = dict(OFFLINE_BASE_ENV)
    if extra:
        applied.update({key: str(value) for key, value in extra.items()})
    target.update(applied)
    return {key: ("<empty>" if not str(value).strip() else "<set>") for key, value in applied.items()}


def credential_source_report(*, root: Path | None = None) -> list[dict[str, object]]:
    """Describe credential provenance without exposing values."""
    from dotenv import dotenv_values

    root = root or Path(__file__).resolve().parents[1]
    file_vals = dotenv_values(root / ".env")
    reports: list[dict[str, object]] = []
    for key in CREDENTIAL_KEYS:
        proc = os.environ.get(key)
        file_val = file_vals.get(key)
        reports.append(
            {
                "key": key,
                "process_present": proc is not None,
                "process_nonempty": bool((proc or "").strip()),
                "dotenv_file_exists": (root / ".env").is_file(),
                "dotenv_nonempty": bool((file_val or "").strip()),
                "dotenv_length": len((file_val or "").strip()),
                "effective_source": (
                    "process_env"
                    if proc is not None and str(proc).strip()
                    else "process_empty_blocks_dotenv"
                    if proc is not None
                    else "project_dotenv"
                    if (file_val or "").strip()
                    else "unset"
                ),
            }
        )
    return reports
