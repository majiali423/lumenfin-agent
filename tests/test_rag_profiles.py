from __future__ import annotations

import unittest

from lumenfin.rag.profiles import (
    CI_RAG_ENV,
    SHOWCASE_RAG_ENV,
    apply_ci_rag_env,
    apply_showcase_rag_env,
)


class RagProfilesTestCase(unittest.TestCase):
    def test_ci_profile_overwrites(self) -> None:
        env = {
            "MAS_RAG_INDEX_MODE": "async_on_upload",
            "MAS_EMBEDDING_PROVIDER": "dashscope",
            "MAS_EMBEDDING_DIMENSION": "1024",
        }
        apply_ci_rag_env(env)
        self.assertEqual(env["MAS_RAG_INDEX_MODE"], "sync_on_run")
        self.assertEqual(env["MAS_EMBEDDING_PROVIDER"], "deterministic")
        self.assertEqual(env["MAS_EMBEDDING_DIMENSION"], "384")
        self.assertEqual(env["MAS_RAG_RERANK_ENABLED"], "true")

    def test_showcase_fills_missing_only(self) -> None:
        env = {"MAS_EMBEDDING_PROVIDER": "fastembed"}
        apply_showcase_rag_env(env, overwrite=False)
        self.assertEqual(env["MAS_EMBEDDING_PROVIDER"], "fastembed")
        self.assertEqual(env["MAS_RAG_INDEX_MODE"], SHOWCASE_RAG_ENV["MAS_RAG_INDEX_MODE"])
        self.assertEqual(env["MAS_MILVUS_URI"], "data/milvus_lite_dashscope.db")
        self.assertEqual(env["MAS_RAG_RERANK_ENABLED"], "true")

    def test_profiles_disagree_on_index_and_embedder(self) -> None:
        self.assertNotEqual(CI_RAG_ENV["MAS_RAG_INDEX_MODE"], SHOWCASE_RAG_ENV["MAS_RAG_INDEX_MODE"])
        self.assertNotEqual(CI_RAG_ENV["MAS_EMBEDDING_PROVIDER"], SHOWCASE_RAG_ENV["MAS_EMBEDDING_PROVIDER"])


if __name__ == "__main__":
    unittest.main()
