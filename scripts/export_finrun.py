from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.finrun import export_finrun_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a LumenFin *_state.json artifact as FinRun JSON.")
    parser.add_argument("state_json", help="Path to a LumenFin exported *_state.json file.")
    parser.add_argument("--out", required=True, help="Path to write the FinRun JSON artifact.")
    args = parser.parse_args()

    state_path = Path(args.state_json)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    finrun = export_finrun_state(state)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(finrun, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"WROTE {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
