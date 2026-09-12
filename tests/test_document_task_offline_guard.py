from __future__ import annotations

import hashlib
import os
import socket
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from lumenfin.eval.offline_guard import (
    EVAL_OFFLINE_PROCESS_ENV,
    OfflineNetworkError,
    apply_eval_offline_process_env,
    install_outbound_block,
    reset_blocked_attempts,
    uninstall_outbound_block,
)


class OfflineGuardTestCase(unittest.TestCase):
    def test_process_overlay_does_not_write_dotenv(self) -> None:
        root = Path(__file__).resolve().parents[1]
        env_path = root / ".env"
        before = hashlib.sha256(env_path.read_bytes()).hexdigest() if env_path.is_file() else None
        fake = {
            "DEEPSEEK_API_KEY": "placeholder-not-a-secret",
            "DASHSCOPE_API_KEY": "placeholder-not-a-secret",
            "LANGSMITH_API_KEY": "placeholder-not-a-secret",
            "MAS_LANGSMITH_TRACING": "true",
            "LANGCHAIN_TRACING_V2": "true",
            "MAS_EMBEDDING_PROVIDER": "dashscope",
            "MAS_RAG_RERANK_PROVIDER": "qwen3",
            "DASHSCOPE_RERANK_BASE_URL": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        with patch.dict(os.environ, fake, clear=False):
            report = apply_eval_offline_process_env()
            self.assertEqual(os.environ.get("DEEPSEEK_API_KEY"), "")
            self.assertEqual(os.environ.get("DASHSCOPE_API_KEY"), "")
            self.assertEqual(os.environ.get("LANGSMITH_API_KEY"), "")
            self.assertEqual(os.environ.get("MAS_EMBEDDING_PROVIDER"), "deterministic")
            self.assertEqual(os.environ.get("MAS_RAG_RERANK_PROVIDER"), "lexical")
            self.assertEqual(os.environ.get("MAS_LANGSMITH_TRACING"), "false")
            self.assertIn("DEEPSEEK_API_KEY", report)
        after = hashlib.sha256(env_path.read_bytes()).hexdigest() if env_path.is_file() else None
        self.assertEqual(before, after)

    def test_outbound_block_rejects_provider_hosts(self) -> None:
        reset_blocked_attempts()
        install_outbound_block()
        try:
            with self.assertRaises(OfflineNetworkError):
                socket.getaddrinfo("api.deepseek.com", 443)
            with self.assertRaises(OfflineNetworkError):
                socket.create_connection(("dashscope.aliyuncs.com", 443), timeout=0.2)
            with self.assertRaises(OfflineNetworkError):
                socket.getaddrinfo("api.smith.langchain.com", 443)
            info = socket.getaddrinfo("127.0.0.1", 9)
            self.assertTrue(info)
        finally:
            uninstall_outbound_block()

    def test_offline_run_priority_over_provider_env(self) -> None:
        from scripts.offline_env import apply_offline_env

        apply_offline_env()
        apply_eval_offline_process_env()
        from lumenfin.eval.baselines import run_b1, score_run
        from lumenfin.eval.document_tasks import tasks_for

        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p21-narrative-refuse-oi")
        root = Path("test_artifacts") / f"offline-guard-{uuid4().hex[:8]}"
        root.mkdir(parents=True, exist_ok=True)
        with patch.dict(
            os.environ,
            {
                **EVAL_OFFLINE_PROCESS_ENV,
                "DEEPSEEK_API_KEY": "",
                "DASHSCOPE_API_KEY": "placeholder-not-a-secret",
                "MAS_RAG_RERANK_PROVIDER": "qwen3",
            },
            clear=False,
        ):
            apply_eval_offline_process_env()
            scored = score_run(task, run_b1(task, root=root / "b1", offline=True))
        self.assertEqual(scored["result_class"], "offline_local_fallback")
        self.assertFalse(scored["accuracy_eligible"])
        self.assertFalse(scored["retrieval_production_equivalent"])
        self.assertIn("gold", scored)
