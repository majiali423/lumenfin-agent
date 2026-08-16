"""Stable FinanceBench evaluation constants."""

from __future__ import annotations

EXPECTED_OPEN_SOURCE_QUESTIONS = 150
SPLIT_DEV_SIZE = 50
SPLIT_TEST_SIZE = 100
SPLIT_SALT = "lumenfin-financebench-split-v1"
SPLIT_VERSION = "v1"
DEFAULT_CHUNK_CHARS = 900
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 10
DEFAULT_RERANK_CANDIDATES = 20
DEFAULT_BM25_RRF_WEIGHT = 1.1
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_BOOTSTRAP_SEED = 20260816
TUNABLE_SPLITS = frozenset({"dev"})
FROZEN_SPLITS = frozenset({"test"})
RETRIEVAL_MODES = ("bm25", "dense", "hybrid", "hybrid-qwen3")
REMOTE_MODES = frozenset({"hybrid-qwen3"})
REMOTE_EMBEDDING_PROVIDERS = frozenset({"dashscope", "aliyun", "alibaba", "通义"})
SCHEMA_VERSION = "financebench_eval.v1"
PAGE_K_VALUES = (1, 3, 5, 10)
CHUNK_K_VALUES = (5, 10, 20)
EVAL_SESSION_ID = "financebench-eval-v1"
EVAL_COMPANY_TAG = "FinanceBenchEval"
EVAL_COLLECTION = "financebench_eval_v1"
DEFAULT_SOURCE_RELATIVE = "data/external/financebench-src"
EVAL_ANCHOR_DOCUMENT = {
    "document_id": "financebench-eval-anchor",
    "filename": "financebench-eval-anchor",
    "pages": [],
    "issuer_companies": ["FinanceBenchEval"],
    "detected_companies": ["FinanceBenchEval"],
}
