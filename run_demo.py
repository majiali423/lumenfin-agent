from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin import LumenFinAgentSystem
from lumenfin.config import AppConfig
from lumenfin.reporting import export_run_artifacts
from lumenfin.stdio import configure_stdio_utf8


def main() -> None:
    configure_stdio_utf8()
    for noisy in ("grpc", "grpc._server", "pymilvus"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    parser = argparse.ArgumentParser(description="Run the LumenFin agent demo.")
    parser.add_argument(
        "--query",
        default="",
        help="User query (see docs/MY_TEST_QUERIES.md for examples).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full final state as JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store generated markdown and JSON artifacts.",
    )
    parser.add_argument(
        "--thread-id",
        default="demo-thread",
        help="Thread identifier used by the LangGraph checkpointer.",
    )
    args = parser.parse_args()
    if not args.query.strip():
        parser.error("Missing --query. Examples: docs/MY_TEST_QUERIES.md")

    config = AppConfig.from_env()
    output_dir = Path(args.output_dir) if args.output_dir else config.output_dir

    app = LumenFinAgentSystem()
    result = app.run(args.query, thread_id=args.thread_id)
    artifacts = export_run_artifacts(result, output_dir=output_dir, thread_id=args.thread_id)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            result.get("final_report")
            or f"(no final_report; workflow_status={result.get('workflow_status')})"
        )
        print(f"\n[LLM] backend={result.get('llm_backend', 'unknown')}")
        print(f"[Workflow] status={result.get('workflow_status')}")
        if result.get("missing_fields"):
            print(f"[HITL] missing_fields={result.get('missing_fields')}")
            print(f"[HITL] questions={result.get('clarification_questions')}")
        print(f"\n[Artifacts] report={artifacts.get('report_path')}")
        print(f"[Artifacts] audit={artifacts.get('audit_path')}")
        print(f"[Artifacts] state={artifacts.get('state_path')}")
        print(f"[Artifacts] manifest={artifacts.get('manifest_path')}")


if __name__ == "__main__":
    main()
