from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.eval.holdout import HoldoutError
from lumenfin.eval.holdout.ledger_e2e import build_generation_prompt
from lumenfin.eval.holdout.ledger_parent_return import (
    build_parent_page_hits,
    parent_prompt_char_cap,
)


def _load_cli():
    path = ROOT / "scripts" / "run_ledger_public_dev_parent_page_e2e.py"
    spec = importlib.util.spec_from_file_location(
        "run_ledger_public_dev_parent_page_e2e",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load parent-page e2e CLI")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_sealer():
    path = ROOT / "scripts" / "seal_ledger_public_dev_parent_page_e2e.py"
    spec = importlib.util.spec_from_file_location(
        "seal_ledger_public_dev_parent_page_e2e",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load parent-page e2e sealer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeLLM:
    def __init__(self, payload: str) -> None:
        self.payload = payload
        self.last_attempts = 1
        self.calls = 0

    def mark_usage_start(self) -> None:
        return None

    def usage_since_mark(self) -> dict[str, int]:
        return {"prompt_tokens": 10, "completion_tokens": 5}

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
    ) -> str:
        self.calls += 1
        if temperature != 0.0:
            raise AssertionError("generation must be locked to temperature 0")
        return self.payload


class LedgerParentPageE2ETests(unittest.TestCase):
    def test_plan_phase_rejects_allow_remote(self) -> None:
        cli = _load_cli()
        with tempfile.TemporaryDirectory() as tmp:
            code = cli.main(
                [
                    "--phase",
                    "plan",
                    "--parquet-path",
                    str(Path(tmp) / "missing.parquet"),
                    "--manifest",
                    str(Path(tmp) / "manifest.json"),
                    "--split-salt",
                    "salt",
                    "--candidate-dir",
                    tmp,
                    "--paired-aggregate",
                    str(Path(tmp) / "qwen3.json"),
                    "--paired-per-case",
                    str(Path(tmp) / "qwen3.jsonl"),
                    "--e2e-aggregate",
                    str(Path(tmp) / "e2e.json"),
                    "--suffix-aggregate",
                    str(Path(tmp) / "suffix.json"),
                    "--baseline-aggregate",
                    str(Path(tmp) / "base.json"),
                    "--baseline-per-case",
                    str(Path(tmp) / "base.jsonl"),
                    "--prerank-aggregate",
                    str(Path(tmp) / "pre.json"),
                    "--output-dir",
                    str(Path(tmp) / "out"),
                    "--allow-remote",
                ]
            )
        self.assertEqual(code, 2)

    def test_parent_prompt_keeps_full_page_not_4000(self) -> None:
        page = "Accounts payable 1,223,000. " + ("x" * 5000)
        hits = build_parent_page_hits(
            [{"chunk_id": "c0", "document_id": "NYSE_MLR_2017/page_0004"}],
            {"NYSE_MLR_2017/page_0004": page},
        )
        cap = parent_prompt_char_cap(hits)
        self.assertGreater(cap, 4000)
        prompt = build_generation_prompt(
            query_text="What is accounts payable?",
            hits=hits,
            max_document_chars=cap,
        )
        self.assertIn("1,223,000", prompt)
        self.assertIn("x" * 5000, prompt)

    def test_missing_qwen3_chunk_fails_closed(self) -> None:
        cli = _load_cli()
        with self.assertRaisesRegex(HoldoutError, "outside the frozen pool"):
            cli._chunk_hits(
                {"hits": [{"chunk_id": "other", "document_id": "doc"}]},
                [{"chunk_id": "wanted", "document_id": "doc"}],
            )

    def test_generate_uses_supplied_hits_without_rerank(self) -> None:
        cli = _load_cli()
        llm = _FakeLLM(
            '{"value": 1223000, "cited_chunk_ids": ["NYSE_MLR_2017/page_0004"], "abstain": false}'
        )
        hits = build_parent_page_hits(
            [{"chunk_id": "c0", "document_id": "NYSE_MLR_2017/page_0004"}],
            {"NYSE_MLR_2017/page_0004": "Accounts payable 1,223,000"},
        )
        result = cli._run_generate(
            arm="parent_page",
            query={
                "query_text": "What is accounts payable?",
                "qrels": [{"doc_id": "NYSE_MLR_2017/page_0004", "relevance": 1}],
            },
            hits=hits,
            gold_value=1_223_000.0,
            llm=llm,
            max_document_chars=parent_prompt_char_cap(hits),
        )
        self.assertEqual(llm.calls, 1)
        self.assertTrue(result["numeric_match"])
        self.assertEqual(result["arm"], "parent_page")

    def test_completed_row_hash_mismatch_fails_closed(self) -> None:
        cli = _load_cli()
        row = {
            "query_id": "q1",
            "run_identity_sha256": "run",
            "shared_candidate_identity_sha256": "cand",
            "arms": {"chunk": {}, "parent_page": {}},
            "row_sha256": "wrong",
        }
        with self.assertRaisesRegex(HoldoutError, "identity mismatch"):
            cli._validate_completed(
                [row],
                candidate_by_id={"q1": {"candidate_identity_sha256": "cand"}},
                run_identity="run",
            )

    def test_sealer_rejects_non_tracked_output(self) -> None:
        sealer = _load_sealer()
        with tempfile.TemporaryDirectory() as tmp:
            code = sealer.main(
                [
                    "--aggregate",
                    str(Path(tmp) / "aggregate.json"),
                    "--per-case",
                    str(Path(tmp) / "per_case.jsonl"),
                    "--plan",
                    str(Path(tmp) / "plan.json"),
                    "--output",
                    str(Path(tmp) / "wrong.json"),
                ]
            )
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
