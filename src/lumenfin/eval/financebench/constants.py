"""Stable FinanceBench evaluation constants.

Split assignment is keyed by ``financebench_id`` and a versioned salt so that
reordering the JSONL cannot move a case between development and held-out test.
"""

from __future__ import annotations

EXPECTED_OPEN_SOURCE_QUESTIONS = 150
SPLIT_DEV_SIZE = 50
SPLIT_TEST_SIZE = 100
SPLIT_SALT = "lumenfin-financebench-split-v1"
SPLIT_VERSION = "v1"

# FinanceBench README: evidence_page_num is ZERO-indexed.
# LumenFin chunking enumerates PDF pages from 1.
PAGE_INDEX_BASE = "zero"
LUMENFIN_PAGE_BASE = "one"

DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 10
DEFAULT_RERANK_CANDIDATES = 20
DEFAULT_BM25_RRF_WEIGHT = 1.1
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 20260816

# confirmation-50 is the remaining unseen split; no split is tunable.
TUNABLE_SPLITS = frozenset()
FROZEN_SPLITS = frozenset({"test", "dev", "confirmation", "all"})
CLI_SPLITS = ("dev", "test", "all", "confirmation")
SPLIT_ALIASES = {"confirmation": "dev"}

RETRIEVAL_MODES = ("bm25", "dense", "hybrid", "hybrid-qwen3")
CLI_MODES = RETRIEVAL_MODES + ("all",)
INDEX_SCOPES = ("company", "corpus")
REMOTE_MODES = frozenset({"hybrid-qwen3"})
REMOTE_EMBEDDING_PROVIDERS = frozenset({"dashscope", "aliyun", "alibaba", "通义"})
DEFAULT_DASHSCOPE_EMBEDDING_DIM = 1024

SCHEMA_VERSION = "financebench_eval.v1"
FALLBACK_MANIFEST_NAME = "pdf_fallback.json"
FROZEN_CONFIG_HASH = "18a483f604f3a5420264e746d9219e77e3c9bddbd91c5c50252025b40ccb1ee7"

CORPUS_BASELINE = {
    "recorded_at": "2026-08-16",
    "split": "test",
    "split_status": "exposed_test",
    "experiment_role": "exploratory_baseline",
    "index_scope": "corpus",
    "documents": 84,
    "chunks": 52518,
    "embedding_provider": "dashscope",
    "embedding_model": "text-embedding-v4",
    "embedding_dimension": 1024,
    "top_k": 10,
    "bm25_rrf_weight": 1.1,
    "worktree": "dirty",
    "page": {
        "bm25": {
            "hit_at_1": 0.11,
            "hit_at_5": 0.21,
            "hit_at_10": 0.30,
            "mrr": 0.1603,
            "ndcg_at_10": 0.1763,
        },
        "dense": {
            "hit_at_1": 0.12,
            "hit_at_5": 0.37,
            "hit_at_10": 0.60,
            "mrr": 0.2465,
            "ndcg_at_10": 0.3099,
        },
        "hybrid": {
            "hit_at_1": 0.13,
            "hit_at_5": 0.23,
            "hit_at_10": 0.38,
            "mrr": 0.1836,
            "ndcg_at_10": 0.2137,
        },
        "hybrid-qwen3": {
            "hit_at_1": 0.19,
            "hit_at_5": 0.47,
            "hit_at_10": 0.58,
            "mrr": 0.3044,
            "ndcg_at_10": 0.3479,
        },
    },
}
