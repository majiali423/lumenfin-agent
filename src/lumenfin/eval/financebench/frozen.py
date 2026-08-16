"""Machine-checkable FinanceBench confirmation-50 frozen configuration.

Canonical hash
--------------
UTF-8 ``json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))``
then SHA-256 hex digest.

Excluded from the digest: ``config_hash`` and timestamp keys
(``proposed_at``, ``executed_at``, ``frozen_at``, ``timestamp``).

The published digest ``18a483f6…`` was generated from the original candidate
payload after dropping only ``config_hash`` and ``proposed_at``. That payload
**includes** ``notes`` plus identity fields (``name``, ``experiment_role``,
``held_out_claim``, ``split_for_confirmation``, ``pdf_parser``). Dropping
``notes`` yields a different digest; this module refuses to mint a replacement.
``notes`` is therefore frozen hash material for this configuration, not a
free-form comment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ...rag.dashscope_defaults import resolved_dashscope_embedding_model
from ...rag.rerank import DEFAULT_RERANK_INSTRUCT
from .constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_CHUNK_CHARS,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_DASHSCOPE_EMBEDDING_DIM,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_TOP_K,
    REMOTE_EMBEDDING_PROVIDERS,
)
from .reporting import git_snapshot, sha256_file
from .split import canonicalize_split

PUBLISHED_CONFIG_HASH = "18a483f604f3a5420264e746d9219e77e3c9bddbd91c5c50252025b40ccb1ee7"
DEFAULT_FROZEN_CONFIG_PATH = Path("data") / "eval_rag" / "financebench" / "frozen_config.json"
HASH_SKIP_KEYS = frozenset(
    {
        "config_hash",
        "proposed_at",
        "executed_at",
        "frozen_at",
        "timestamp",
    }
)
CONFIRMATION_MODE = "hybrid-qwen3"
CONFIRMATION_SCOPE = "company"


class FrozenConfigError(ValueError):
    """Raised when confirmation-50 is not locked to the published frozen config."""


@dataclass(frozen=True)
class FrozenConfig:
    path: Path
    payload: dict[str, Any]
    config_hash: str

    @property
    def retrieval_mode(self) -> str:
        return str(self.payload.get("retrieval_mode") or "")

    @property
    def index_scope(self) -> str:
        return str(self.payload.get("index_scope") or "")

    @property
    def embedding_provider(self) -> str:
        return str(self.payload.get("embedding_provider") or "")

    @property
    def embedding_model(self) -> str:
        return str(self.payload.get("embedding_model") or "")

    @property
    def embedding_dimension(self) -> int:
        return int(self.payload.get("embedding_dimension") or 0)

    @property
    def chunk_size(self) -> int:
        return int(self.payload.get("chunk_size") or 0)

    @property
    def chunk_overlap(self) -> int:
        return int(self.payload.get("chunk_overlap") or 0)

    @property
    def bm25_rrf_weight(self) -> float:
        return float(self.payload.get("bm25_rrf_weight") or 0.0)

    @property
    def top_k(self) -> int:
        return int(self.payload.get("top_k") or 0)

    @property
    def rerank_candidates(self) -> int:
        return int(self.payload.get("rerank_candidates") or 0)

    @property
    def rerank_provider(self) -> str:
        return str(self.payload.get("rerank_provider") or "")

    @property
    def rerank_model(self) -> str:
        return str(self.payload.get("rerank_model") or "")

    @property
    def rerank_instruct(self) -> str:
        return str(self.payload.get("rerank_instruct") or "")

    @property
    def query_rewriting(self) -> bool:
        return bool(self.payload.get("query_rewriting"))

    @property
    def dataset_hash(self) -> str:
        return str(self.payload.get("dataset_hash") or "")


def is_confirmation_split(split: str) -> bool:
    raw = str(split or "").strip().lower()
    if raw == "confirmation":
        return True
    return canonicalize_split(raw) == "dev"


def hash_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in HASH_SKIP_KEYS}


def compute_config_hash(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        hash_material(payload),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    from hashlib import sha256

    return sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_instruct(text: str) -> str:
    return " ".join(str(text or "").split())


def load_frozen_config(path: str | Path) -> FrozenConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FrozenConfigError(f"frozen config file is missing: {config_path}")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FrozenConfigError(f"frozen config is not valid JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise FrozenConfigError("frozen config must be a JSON object")
    stored = str(payload.get("config_hash") or "").strip()
    computed = compute_config_hash(payload)
    if not stored:
        raise FrozenConfigError("frozen config is missing config_hash")
    if stored != computed:
        raise FrozenConfigError(
            "frozen config_hash does not match canonical digest of the payload; "
            "refusing to mint a replacement hash"
        )
    if stored != PUBLISHED_CONFIG_HASH:
        raise FrozenConfigError(
            "frozen config_hash is not the published confirmation digest"
        )
    return FrozenConfig(path=config_path.resolve(), payload=payload, config_hash=stored)


def resolved_embedding_model(provider: str) -> str:
    name = provider.strip().lower()
    if name in REMOTE_EMBEDDING_PROVIDERS:
        return resolved_dashscope_embedding_model()
    return "deterministic-hash"


def resolved_embedding_dimension(provider: str, dimension: int) -> int:
    name = provider.strip().lower()
    env_dim = os.getenv("DASHSCOPE_EMBEDDING_DIMENSION")
    if name in REMOTE_EMBEDDING_PROVIDERS:
        if env_dim:
            return int(env_dim)
        if dimension <= 0 or dimension == 384:
            return DEFAULT_DASHSCOPE_EMBEDDING_DIM
        return int(dimension)
    return 384 if dimension <= 0 else int(dimension)


def resolved_rerank_model() -> str:
    return (os.getenv("DASHSCOPE_RERANK_MODEL") or "qwen3-rerank").strip()


def resolved_rerank_instruct() -> str:
    return (os.getenv("MAS_RAG_RERANK_INSTRUCT") or DEFAULT_RERANK_INSTRUCT).strip()


def provenance(config: FrozenConfig, *, verified: bool = True) -> dict[str, Any]:
    return {
        "frozen_config_hash": config.config_hash,
        "frozen_config_path": str(config.path),
        "frozen_config_verified": bool(verified),
    }


def enforce_confirmation_lock(
    *,
    split: str,
    mode: str,
    index_scope: str,
    embedding_provider: str,
    embedding_dimension: int,
    top_k: int,
    bm25_rrf_weight: float = DEFAULT_BM25_RRF_WEIGHT,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    chunk_size: int = DEFAULT_CHUNK_CHARS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    query_rewriting: bool = False,
    limit: int | None,
    frozen_config_path: str | Path | None,
    confirm_held_out: bool,
    repo_root: str | Path,
    dataset_dir: str | Path | None = None,
    require_clean_worktree: bool = True,
    verify_dataset_hash: bool = True,
) -> FrozenConfig:
    """Fail closed before dataset parse, indexing, or remote provider calls."""
    if not is_confirmation_split(split):
        raise FrozenConfigError("internal: confirmation lock called for a non-confirmation split")
    if not confirm_held_out:
        raise FrozenConfigError(
            "confirmation/dev requires explicit --confirm-held-out; "
            "this split is the unseen confirmation-50"
        )
    if not frozen_config_path:
        raise FrozenConfigError("confirmation/dev requires --frozen-config")
    if limit is not None:
        raise FrozenConfigError("confirmation/dev cannot be combined with --limit")
    if mode == "all":
        raise FrozenConfigError("confirmation/dev forbids --mode all; only hybrid-qwen3 is frozen")
    config = load_frozen_config(frozen_config_path)

    mismatches: list[str] = []
    if mode != config.retrieval_mode:
        mismatches.append("mode")
    if index_scope != config.index_scope:
        mismatches.append("index_scope")
    if embedding_provider.strip().lower() != config.embedding_provider:
        mismatches.append("embedding_provider")
    actual_dim = resolved_embedding_dimension(embedding_provider, embedding_dimension)
    if actual_dim != config.embedding_dimension:
        mismatches.append("embedding_dimension")
    if int(top_k) != config.top_k:
        mismatches.append("top_k")
    if abs(float(bm25_rrf_weight) - config.bm25_rrf_weight) > 1e-12:
        mismatches.append("bm25_rrf_weight")
    if int(rerank_candidates) != config.rerank_candidates:
        mismatches.append("rerank_candidates")
    if int(chunk_size) != config.chunk_size:
        mismatches.append("chunk_size")
    if int(chunk_overlap) != config.chunk_overlap:
        mismatches.append("chunk_overlap")
    if bool(query_rewriting) != config.query_rewriting:
        mismatches.append("query_rewriting")
    actual_embed_model = resolved_embedding_model(embedding_provider)
    if actual_embed_model != config.embedding_model:
        mismatches.append("embedding_model")
    if resolved_rerank_model() != config.rerank_model:
        mismatches.append("rerank_model")
    if _normalize_instruct(resolved_rerank_instruct()) != _normalize_instruct(config.rerank_instruct):
        mismatches.append("rerank_instruct")
    if mismatches:
        raise FrozenConfigError(
            "runtime settings do not match the frozen confirmation config: "
            + ", ".join(mismatches)
        )

    snapshot = git_snapshot(Path(repo_root))
    if require_clean_worktree and snapshot.get("worktree_dirty"):
        raise FrozenConfigError(
            "confirmation/dev requires a clean git worktree; commit or stash first"
        )
    if verify_dataset_hash:
        if dataset_dir is None:
            raise FrozenConfigError("confirmation/dev requires a dataset directory to verify dataset_hash")
        from .loader import FinanceBenchLoadError, discover_financebench_paths

        try:
            paths = discover_financebench_paths(dataset_dir)
        except FinanceBenchLoadError as exc:
            raise FrozenConfigError(f"confirmation dataset is not available: {exc}") from exc
        actual_hash = sha256_file(paths.questions_path)
        if actual_hash != config.dataset_hash:
            raise FrozenConfigError("dataset_hash does not match the frozen confirmation config")
    return config
