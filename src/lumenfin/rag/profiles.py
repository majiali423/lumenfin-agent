"""Named RAG env profiles for showcase vs CI.

Showcase / real PDF analysis: async index + DashScope + dedicated 1024-dim Lite DB.
Unit tests / CI: sync_on_run + deterministic (no remote embedding dependency).
"""
from __future__ import annotations

import os
from typing import Mapping, MutableMapping

# Local Lite only — do not enable Milvus Server / Redis worker / BM25 v4 here.
# Production Compose uses lumenfin_chunks_v4_bm25 (see BM25_CUTOVER.md).
SHOWCASE_RAG_ENV: dict[str, str] = {
    "MAS_RAG_ENABLED": "true",
    "MAS_RAG_INDEX_MODE": "async_on_upload",
    "MAS_EMBEDDING_PROVIDER": "dashscope",
    "MAS_EMBEDDING_DIMENSION": "1024",
    "DASHSCOPE_EMBEDDING_DIMENSION": "1024",
    "MAS_MILVUS_URI": "data/milvus_lite_dashscope.db",
    "MAS_MILVUS_COLLECTION": "lumenfin_chunks_ds",
    "MAS_MILVUS_ISOLATE": "true",
    "MAS_RAG_TOP_K": "5",
    "MAS_RAG_RERANK_ENABLED": "true",
    "MAS_RAG_RERANK_CANDIDATES": "20",
    "MAS_RAG_DEGRADE_ON_VECTOR_ERROR": "true",
    "MAS_RAG_SANITIZE_HITS": "true",
}

CI_RAG_ENV: dict[str, str] = {
    "MAS_RAG_ENABLED": "true",
    "MAS_RAG_INDEX_MODE": "sync_on_run",
    "MAS_EMBEDDING_PROVIDER": "deterministic",
    "MAS_EMBEDDING_DIMENSION": "384",
    "MAS_MILVUS_URI": "data/milvus_lite_ci.db",
    "MAS_MILVUS_COLLECTION": "lumenfin_chunks_ci",
    "MAS_MILVUS_ISOLATE": "true",
    "MAS_RAG_TOP_K": "5",
    "MAS_RAG_RERANK_ENABLED": "true",
    "MAS_RAG_RERANK_CANDIDATES": "20",
    "MAS_RAG_DEGRADE_ON_VECTOR_ERROR": "true",
    "MAS_RAG_SANITIZE_HITS": "true",
    # Keep async index worker / Server / remote embed keys out of CI
    "MAS_REDIS_URL": "",
    "DASHSCOPE_API_KEY": "",
    "DASHSCOPE_EMBEDDING_DIMENSION": "384",
}


def apply_rag_profile(
    profile: Mapping[str, str],
    environ: MutableMapping[str, str] | None = None,
    *,
    overwrite: bool = True,
) -> MutableMapping[str, str]:
    """Apply a RAG profile into an env mapping (defaults to os.environ)."""
    target: MutableMapping[str, str] = os.environ if environ is None else environ
    for key, value in profile.items():
        if overwrite or key not in target or not str(target.get(key, "")).strip():
            target[key] = value
    return target


def apply_ci_rag_env(environ: MutableMapping[str, str] | None = None) -> MutableMapping[str, str]:
    return apply_rag_profile(CI_RAG_ENV, environ, overwrite=True)


def apply_showcase_rag_env(
    environ: MutableMapping[str, str] | None = None,
    *,
    overwrite: bool = False,
) -> MutableMapping[str, str]:
    """Showcase profile. Default overwrite=False so an explicit .env wins when present."""
    return apply_rag_profile(SHOWCASE_RAG_ENV, environ, overwrite=overwrite)
