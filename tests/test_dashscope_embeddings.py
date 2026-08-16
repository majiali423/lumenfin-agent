from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.provider_resilience import InvalidProviderResponseError
from lumenfin.rag.dashscope_defaults import DEFAULT_DASHSCOPE_EMBEDDING_MODEL
from lumenfin.rag.embeddings import (
    DashScopeEmbeddingProvider,
    build_embedding_provider,
)


class DashScopeEmbeddingTestCase(unittest.TestCase):
    def test_build_provider_aliases_require_api_key(self) -> None:
        import os
        from unittest.mock import patch

        for name in ("dashscope", "aliyun", "alibaba"):
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": ""}, clear=False):
                with self.assertRaises(ValueError) as ctx:
                    build_embedding_provider(name, dimension=1024)
                self.assertIn("DASHSCOPE_API_KEY", str(ctx.exception))

    def test_embed_batches_and_orders_by_index(self) -> None:
        mock_client = MagicMock()
        response = MagicMock()
        response.is_error = False
        response.raise_for_status = MagicMock()
        vec_a = [1.0] + [0.0] * 1023
        vec_b = [0.0, 1.0] + [0.0] * 1022
        response.json.return_value = {
            "data": [
                {"index": 1, "embedding": vec_b},
                {"index": 0, "embedding": vec_a},
            ]
        }
        mock_client.post.return_value = response

        provider = DashScopeEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-v3",
            dimension=1024,
            client=mock_client,
        )
        out = provider.embed(["alpha", "beta"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], 1.0)
        self.assertEqual(out[1][1], 1.0)
        self.assertEqual(mock_client.post.call_count, 1)
        args, kwargs = mock_client.post.call_args
        self.assertIn("/embeddings", args[0])
        self.assertEqual(kwargs["json"]["dimensions"], 1024)
        self.assertEqual(kwargs["json"]["input"], ["alpha", "beta"])

    def test_rejects_bad_dimension(self) -> None:
        with self.assertRaises(ValueError):
            DashScopeEmbeddingProvider(api_key="sk-test", dimension=3)

    def test_error_body_is_redacted_and_does_not_use_raise_for_status(self) -> None:
        mock_client = MagicMock()
        response = MagicMock()
        response.is_error = True
        response.status_code = 400
        response.text = (
            '{"message":"quota","api_key":"sk-secretvalue","endpoint":'
            '"https://dashscope.aliyuncs.com/api/v1/embeddings"}'
        )
        response.raise_for_status.side_effect = AssertionError("must not dump raw body")
        mock_client.post.return_value = response
        provider = DashScopeEmbeddingProvider(
            api_key="sk-test",
            model="text-embedding-v4",
            dimension=1024,
            client=mock_client,
        )
        with self.assertRaises(InvalidProviderResponseError) as ctx:
            provider.embed(["alpha"])
        message = str(ctx.exception)
        self.assertIn("HTTP 400", message)
        self.assertNotIn("sk-secretvalue", message)
        self.assertNotIn("https://dashscope.aliyuncs.com", message)

    def test_build_provider_defaults_to_v4_when_model_env_unset(self) -> None:
        import os
        from unittest.mock import patch

        overlay = {
            key: value
            for key, value in os.environ.items()
            if key not in {"DASHSCOPE_EMBEDDING_MODEL", "DASHSCOPE_EMBEDDING_DIMENSION"}
        }
        overlay["DASHSCOPE_API_KEY"] = "sk-test"
        with patch.dict(os.environ, overlay, clear=True):
            provider = build_embedding_provider("dashscope", dimension=1024)
        self.assertEqual(provider.inner._model, DEFAULT_DASHSCOPE_EMBEDDING_MODEL)
        self.assertEqual(DEFAULT_DASHSCOPE_EMBEDDING_MODEL, "text-embedding-v4")

    def test_build_provider_honors_explicit_v3_override(self) -> None:
        import os
        from unittest.mock import patch

        overlay = {
            key: value
            for key, value in os.environ.items()
            if key != "DASHSCOPE_EMBEDDING_DIMENSION"
        }
        overlay["DASHSCOPE_API_KEY"] = "sk-test"
        overlay["DASHSCOPE_EMBEDDING_MODEL"] = "text-embedding-v3"
        with patch.dict(os.environ, overlay, clear=True):
            provider = build_embedding_provider("dashscope", dimension=1024)
        self.assertEqual(provider.inner._model, "text-embedding-v3")


if __name__ == "__main__":
    unittest.main()
