from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.financebench_fixtures import make_financebench_tree
from tests.test_financebench_retrieval_eval import FakeRAGStore

from lumenfin.eval.financebench.frozen import (
    PUBLISHED_CONFIG_HASH,
    FrozenConfigError,
    compute_config_hash,
    enforce_confirmation_lock,
    load_frozen_config,
)
from lumenfin.eval.financebench.runner import run_retrieval_eval
from lumenfin.rag.dashscope_defaults import DEFAULT_DASHSCOPE_EMBEDDING_MODEL

FROZEN_PATH = ROOT / "data" / "eval_rag" / "financebench" / "frozen_config.json"
INSTRUCT = (
    "Given a financial due diligence query, retrieve passages that directly answer it. "
    "Prefer the correct company, reporting period, metric, scope, and filing context over "
    "merely topical passages."
)
MATCHING_ENV = {
    "DASHSCOPE_EMBEDDING_MODEL": "text-embedding-v4",
    "DASHSCOPE_EMBEDDING_DIMENSION": "1024",
    "DASHSCOPE_RERANK_MODEL": "qwen3-rerank",
    "MAS_RAG_RERANK_INSTRUCT": INSTRUCT,
}


def _env_with_embedding_model(model: str | None) -> dict[str, str]:
    overlay = dict(os.environ)
    overlay.update(MATCHING_ENV)
    if model is None:
        overlay.pop("DASHSCOPE_EMBEDDING_MODEL", None)
    else:
        overlay["DASHSCOPE_EMBEDDING_MODEL"] = model
    return overlay


def _lock_kwargs(**overrides):
    values = {
        "split": "confirmation",
        "mode": "hybrid-qwen3",
        "index_scope": "company",
        "embedding_provider": "dashscope",
        "embedding_dimension": 1024,
        "top_k": 10,
        "limit": None,
        "frozen_config_path": FROZEN_PATH,
        "confirm_held_out": True,
        "repo_root": ROOT,
        "require_clean_worktree": False,
        "verify_dataset_hash": False,
    }
    values.update(overrides)
    return values


class FinanceBenchFrozenConfigTests(unittest.TestCase):
    def test_published_hash_matches_canonical_payload(self) -> None:
        payload = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
        computed = compute_config_hash(payload)
        self.assertEqual(computed, PUBLISHED_CONFIG_HASH)
        self.assertEqual(payload["config_hash"], PUBLISHED_CONFIG_HASH)
        without_notes = dict(payload)
        without_notes.pop("notes", None)
        self.assertNotEqual(compute_config_hash(without_notes), PUBLISHED_CONFIG_HASH)

    def test_valid_config_passes_without_remote_or_index(self) -> None:
        with patch.dict("os.environ", MATCHING_ENV, clear=False):
            with patch(
                "lumenfin.eval.financebench.frozen.git_snapshot",
                return_value={"worktree_dirty": False, "worktree_status": "clean"},
            ):
                with patch("lumenfin.eval.financebench.runner.prepare_financebench_eval") as prepare:
                    with patch("lumenfin.eval.financebench.runner.build_eval_store") as store:
                        config = enforce_confirmation_lock(**_lock_kwargs())
        self.assertEqual(config.config_hash, PUBLISHED_CONFIG_HASH)
        prepare.assert_not_called()
        store.assert_not_called()

    def test_unset_embedding_model_env_passes_frozen_lock(self) -> None:
        with patch.dict("os.environ", _env_with_embedding_model(None), clear=True):
            with patch(
                "lumenfin.eval.financebench.frozen.git_snapshot",
                return_value={"worktree_dirty": False, "worktree_status": "clean"},
            ):
                config = enforce_confirmation_lock(**_lock_kwargs())
        self.assertEqual(config.config_hash, PUBLISHED_CONFIG_HASH)
        self.assertEqual(DEFAULT_DASHSCOPE_EMBEDDING_MODEL, "text-embedding-v4")

    def test_explicit_v3_embedding_model_fails_frozen_lock(self) -> None:
        with patch.dict("os.environ", _env_with_embedding_model("text-embedding-v3"), clear=True):
            with self.assertRaises(FrozenConfigError) as ctx:
                enforce_confirmation_lock(**_lock_kwargs())
        self.assertIn("embedding_model", str(ctx.exception))

    def test_missing_frozen_file_fails(self) -> None:
        with self.assertRaises(FrozenConfigError) as ctx:
            enforce_confirmation_lock(**_lock_kwargs(frozen_config_path=ROOT / "no-such-frozen.json"))
        self.assertIn("missing", str(ctx.exception).lower())

    def test_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frozen.json"
            payload = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
            payload["config_hash"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FrozenConfigError) as ctx:
                load_frozen_config(path)
            self.assertIn("does not match", str(ctx.exception))

    def test_mode_scope_topk_dimension_model_instruct_mismatch(self) -> None:
        with patch.dict("os.environ", MATCHING_ENV, clear=False):
            with self.assertRaises(FrozenConfigError):
                enforce_confirmation_lock(**_lock_kwargs(mode="hybrid"))
            with self.assertRaises(FrozenConfigError):
                enforce_confirmation_lock(**_lock_kwargs(index_scope="corpus"))
            with self.assertRaises(FrozenConfigError):
                enforce_confirmation_lock(**_lock_kwargs(top_k=5))
            with patch.dict("os.environ", {**MATCHING_ENV, "DASHSCOPE_EMBEDDING_DIMENSION": "768"}):
                with self.assertRaises(FrozenConfigError):
                    enforce_confirmation_lock(**_lock_kwargs())
            with patch.dict("os.environ", {**MATCHING_ENV, "DASHSCOPE_EMBEDDING_MODEL": "text-embedding-v3"}):
                with self.assertRaises(FrozenConfigError) as ctx:
                    enforce_confirmation_lock(**_lock_kwargs())
                self.assertIn("embedding_model", str(ctx.exception))
            with patch.dict("os.environ", {**MATCHING_ENV, "MAS_RAG_RERANK_INSTRUCT": "other instruct"}):
                with self.assertRaises(FrozenConfigError):
                    enforce_confirmation_lock(**_lock_kwargs())

    def test_dev_alias_cannot_bypass_protection(self) -> None:
        with self.assertRaises(FrozenConfigError) as ctx:
            enforce_confirmation_lock(**_lock_kwargs(split="dev", confirm_held_out=False))
        self.assertIn("confirm-held-out", str(ctx.exception))

    def test_limit_is_rejected(self) -> None:
        with self.assertRaises(FrozenConfigError) as ctx:
            enforce_confirmation_lock(**_lock_kwargs(limit=2))
        self.assertIn("limit", str(ctx.exception).lower())

    def test_mode_all_is_rejected(self) -> None:
        with self.assertRaises(FrozenConfigError) as ctx:
            enforce_confirmation_lock(**_lock_kwargs(mode="all"))
        self.assertIn("mode all", str(ctx.exception))

    def test_dirty_worktree_is_rejected(self) -> None:
        with patch.dict("os.environ", MATCHING_ENV, clear=False):
            with patch(
                "lumenfin.eval.financebench.frozen.git_snapshot",
                return_value={"worktree_dirty": True, "worktree_status": "dirty"},
            ):
                with self.assertRaises(FrozenConfigError) as ctx:
                    enforce_confirmation_lock(**_lock_kwargs(require_clean_worktree=True))
        self.assertIn("clean git worktree", str(ctx.exception))

    def test_failure_happens_before_index_and_remote(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            with patch("lumenfin.eval.financebench.runner.prepare_financebench_eval") as prepare:
                with patch("lumenfin.eval.financebench.runner.build_eval_store") as store:
                    with self.assertRaises(FrozenConfigError):
                        run_retrieval_eval(
                            dataset_dir=root / "src",
                            output_dir=root / "out",
                            repo_root=ROOT,
                            split="dev",
                            mode="hybrid-qwen3",
                            embedding_provider="dashscope",
                            embedding_dimension=1024,
                            allow_remote=True,
                            confirm_held_out=False,
                        )
            prepare.assert_not_called()
            store.assert_not_called()

    def test_output_provenance_includes_frozen_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_financebench_tree(root / "src")
            fake_store = FakeRAGStore()
            with patch.dict("os.environ", MATCHING_ENV, clear=False):
                with patch(
                    "lumenfin.eval.financebench.runner.build_eval_store",
                    return_value=fake_store,
                ):
                    with patch(
                        "lumenfin.eval.financebench.retrieval._qwen3_reranker",
                    ) as rerank_factory:
                        rerank_factory.return_value.rerank.side_effect = (
                            lambda query, hits, top_k: (hits[:top_k], {"rerank_mode_suffix": "qwen3_rerank"})
                        )
                        results = run_retrieval_eval(
                            dataset_dir=root / "src",
                            output_dir=root / "out",
                            repo_root=ROOT,
                            split="confirmation",
                            mode="hybrid-qwen3",
                            index_scope="company",
                            top_k=10,
                            embedding_provider="dashscope",
                            embedding_dimension=1024,
                            allow_remote=True,
                            frozen_config_path=FROZEN_PATH,
                            confirm_held_out=True,
                            require_clean_worktree=False,
                            verify_dataset_hash=False,
                            expected_questions=4,
                            require_pdfs=True,
                        )
            env = results["environment"]
            self.assertEqual(env["frozen_config_hash"], PUBLISHED_CONFIG_HASH)
            self.assertTrue(env["frozen_config_verified"])
            self.assertIn("frozen_config.json", str(env["frozen_config_path"]).replace("\\", "/"))
            self.assertIn("lumenfin_commit", env)
            self.assertIn(env["worktree_status"], {"clean", "dirty"})
            manifest = json.loads((root / "out" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["frozen_config_hash"], PUBLISHED_CONFIG_HASH)

    def test_environment_records_effective_embedding_model(self) -> None:
        def _run(model_env: str | None) -> dict:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                make_financebench_tree(root / "src")
                fake_store = FakeRAGStore()
                with patch.dict("os.environ", _env_with_embedding_model(model_env), clear=True):
                    with patch(
                        "lumenfin.eval.financebench.runner.build_eval_store",
                        return_value=fake_store,
                    ):
                        with patch(
                            "lumenfin.eval.financebench.retrieval._qwen3_reranker",
                        ) as rerank_factory:
                            rerank_factory.return_value.rerank.side_effect = (
                                lambda query, hits, top_k: (
                                    hits[:top_k],
                                    {"rerank_mode_suffix": "qwen3_rerank"},
                                )
                            )
                            results = run_retrieval_eval(
                                dataset_dir=root / "src",
                                output_dir=root / "out",
                                repo_root=ROOT,
                                split="test",
                                mode="hybrid-qwen3",
                                index_scope="company",
                                top_k=10,
                                embedding_provider="dashscope",
                                embedding_dimension=1024,
                                allow_remote=True,
                                expected_questions=4,
                                require_pdfs=True,
                            )
                env = results["environment"]
                env_disk = json.loads((root / "out" / "environment.json").read_text(encoding="utf-8"))
                self.assertEqual(env["embedding_model"], env_disk["embedding_model"])
                return env

        unset = _run(None)
        self.assertEqual(unset["embedding_model"], "text-embedding-v4")
        self.assertNotEqual(unset["embedding_model"], "text-embedding-v3")
        explicit_v3 = _run("text-embedding-v3")
        self.assertEqual(explicit_v3["embedding_model"], "text-embedding-v3")


if __name__ == "__main__":
    unittest.main()
