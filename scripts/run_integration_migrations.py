#!/usr/bin/env python3
"""Apply PostgreSQL migrations for Phase 3.2B integration (no second framework)."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

MIGRATIONS = [
    ROOT / "migrations" / "postgresql" / "001_add_workflow_checkpoint_revision.sql",
    ROOT / "migrations" / "postgresql" / "002_add_rag_index_lease.sql",
]


def to_psycopg_url(database_url: str) -> str:
    url = database_url.strip()
    if url.startswith("postgresql+psycopg://"):
        return "postgresql://" + url[len("postgresql+psycopg://") :]
    if url.startswith("postgres+psycopg://"):
        return "postgresql://" + url[len("postgres+psycopg://") :]
    return url


def wait_for_postgres(database_url: str, *, timeout_seconds: float = 90.0) -> None:
    import psycopg

    url = to_psycopg_url(database_url)
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with psycopg.connect(url, connect_timeout=3) as conn:
                conn.execute("SELECT 1")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"PostgreSQL not ready after {timeout_seconds}s: {last_error}")


def bootstrap_tables(database_url: str) -> None:
    """Create current ORM tables when the database is empty."""
    from lumenfin.database import Base, JobRepository

    repo = JobRepository(database_url)
    Base.metadata.create_all(repo.engine)


def apply_sql_files(database_url: str, files: list[Path]) -> list[dict[str, str]]:
    import psycopg

    url = to_psycopg_url(database_url)
    results: list[dict[str, str]] = []
    with psycopg.connect(url) as conn:
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            sql = path.read_text(encoding="utf-8")
            conn.execute(sql)
            conn.commit()
            results.append({"file": str(path.relative_to(ROOT)), "status": "applied"})
            print(f"Applied {path.name}")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Run LumenFin PostgreSQL integration migrations.")
    parser.add_argument(
        "--database-url",
        default=os.getenv("MAS_DATABASE_URL", ""),
        help="SQLAlchemy or libpq PostgreSQL URL",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Do not create ORM tables before ALTER migrations",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Apply migration files N times")
    args = parser.parse_args()
    if not args.database_url:
        print("MAS_DATABASE_URL / --database-url is required", file=sys.stderr)
        return 1
    if args.database_url.startswith("sqlite"):
        print("Refusing to run PostgreSQL migrations against SQLite", file=sys.stderr)
        return 1

    wait_for_postgres(args.database_url)
    if not args.skip_bootstrap:
        bootstrap_tables(args.database_url)
        print("Bootstrapped ORM tables")
    for round_idx in range(max(1, args.repeat)):
        if args.repeat > 1:
            print(f"Migration round {round_idx + 1}/{args.repeat}")
        apply_sql_files(args.database_url, MIGRATIONS)
    print("Migrations complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
