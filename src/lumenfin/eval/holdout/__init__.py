"""Unseen-holdout ranking evaluation helpers; no production RAG changes."""

from .dataset import holdout_file_sha256, load_holdout_questions
from .governance import (
    CONSUMED_SPLITS,
    HOLDOUT_SPLIT,
    HoldoutError,
    resolve_holdout_questions_path,
    validate_holdout_request,
)
from .page_collapse import (
    collapse_to_unique_pages,
    duplicate_page_occupancy,
    page_identity_coverage_top_k,
    unique_pages_top_k,
)
from .ranking import (
    ARM_SPECS,
    RankingArm,
    evaluate_ranking_case,
    prepare_rerank_pool,
    summarize_ranking_cases,
)
from .section_schema import (
    SECTION_METADATA_UNAVAILABLE,
    attach_section_metadata,
    section_metadata_for,
)

__all__ = [
    "ARM_SPECS",
    "CONSUMED_SPLITS",
    "HOLDOUT_SPLIT",
    "HoldoutError",
    "RankingArm",
    "SECTION_METADATA_UNAVAILABLE",
    "attach_section_metadata",
    "collapse_to_unique_pages",
    "duplicate_page_occupancy",
    "evaluate_ranking_case",
    "holdout_file_sha256",
    "load_holdout_questions",
    "page_identity_coverage_top_k",
    "prepare_rerank_pool",
    "resolve_holdout_questions_path",
    "section_metadata_for",
    "summarize_ranking_cases",
    "unique_pages_top_k",
    "validate_holdout_request",
]
