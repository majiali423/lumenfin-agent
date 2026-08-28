#!/usr/bin/env python3
"""Freeze LEDGER public_holdout E2E v2 case selection (identity columns only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.ledger_public_holdout_e2e_v2 import (  # noqa: E402
    SELECTION_PATH,
    HoldoutE2EV2Error,
    atomic_write_json,
    freeze_selection,
)
from lumenfin.stdio import configure_stdio_utf8


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description=(
            "Freeze the deterministic 100-case public_holdout selection using "
            "identity columns only. Does not read query_text, gold, or qrels."
        )
    )
    parser.add_argument("--write", action="store_true", help="Write tracked selection JSON")
    args = parser.parse_args(argv)
    try:
        dest = ROOT / SELECTION_PATH
        if args.write and dest.exists():
            existing = json.loads(dest.read_text(encoding="utf-8"))
            if existing.get("holdout_consumed") is True:
                raise HoldoutE2EV2Error("holdout already consumed; refusing selection rewrite")
        payload = freeze_selection(repo_root=ROOT)
        if args.write:
            if dest.exists():
                existing = json.loads(dest.read_text(encoding="utf-8"))
                if existing.get("holdout_consumed") is True:
                    raise HoldoutE2EV2Error("holdout already consumed; refusing selection rewrite")
                if existing.get("selected_query_ids_sha256") != payload["selected_query_ids_sha256"]:
                    raise HoldoutE2EV2Error("refusing to overwrite a different sealed selection")
            atomic_write_json(dest, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    except HoldoutE2EV2Error as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
