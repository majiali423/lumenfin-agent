from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.evaluation import evaluate_run_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exported agent run states.")
    parser.add_argument("--outputs", default="outputs", help="Directory containing *_state.json files.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of latest states to evaluate.")
    parser.add_argument("--write", action="store_true", help="Write evaluation_report.json and evaluation_report.md.")
    args = parser.parse_args()

    output_dir = Path(args.outputs)
    state_paths = sorted(output_dir.glob("*_state.json"), key=lambda path: path.stat().st_mtime, reverse=True)[: args.limit]
    evaluations = [_evaluate_file(path) for path in state_paths]

    payload = {
        "evaluated_runs": len(evaluations),
        "runs": evaluations,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.write:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "evaluation_report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_dir / "evaluation_report.md").write_text(
            _to_markdown(evaluations),
            encoding="utf-8",
        )


def _evaluate_file(path: Path) -> dict[str, Any]:
    state = json.loads(path.read_text(encoding="utf-8"))
    result = evaluate_run_state(state).to_dict()
    result.update(
        {
            "state_path": str(path),
            "thread_id": state.get("thread_id"),
            "companies": state.get("companies", []),
            "llm_backend": state.get("llm_backend", "unknown"),
        }
    )
    return result


def _to_markdown(evaluations: list[dict[str, Any]]) -> str:
    lines = [
        "# Agent Run Evaluation Report",
        "",
        "This report scores exported agent traces, not just final prose. It is designed to show whether the workflow completed the expected steps, produced the expected report contract, and preserved enough evidence for review.",
        "",
        "| Run | Score | Grade | Backend | Companies |",
        "|-----|-------|-------|---------|-----------|",
    ]
    for item in evaluations:
        lines.append(
            f"| {item.get('thread_id') or Path(item['state_path']).name} | {item['score']} | {item['grade']} | "
            f"{item.get('llm_backend', 'unknown')} | {', '.join(item.get('companies', []))} |"
        )
    lines.append("")
    for item in evaluations:
        lines.append(f"## {item.get('thread_id') or Path(item['state_path']).name}")
        lines.append("")
        lines.append(f"- Score: {item['score']} ({item['grade']})")
        lines.append(f"- State: `{item['state_path']}`")
        lines.append("- Recommendations:")
        for recommendation in item["recommendations"]:
            lines.append(f"  - {recommendation}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
