"""LEDGER public/dev structured-citation shadow evaluation harness.

Binds to the sealed public/dev chain. Does not open public_holdout, does not
rewrite sealed aggregates, and does not retune production retrieval.

A future official run is an exposed public/dev shadow only:
held_out=false, not a product-accuracy claim, not a LEDGER benchmark, and
not rc5. This module implements tools, preflight, and resume. It does not
itself authorize a paid remote run.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import subprocess
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from ..env_bootstrap import describe_credential_sources
from ..llm import LLMSettings
from ..provider_resilience import classify_provider_exception
from ..rag.dashscope_defaults import DEFAULT_DASHSCOPE_EMBEDDING_MODEL
from ..rag.rerank import DEFAULT_RERANK_INSTRUCT, LexicalReranker
from ..structured_answer import (
    CITATION_PATH_VALIDATION_FAILED,
    CITATION_SOURCE_STRUCTURED,
    CITATION_SOURCE_UNAVAILABLE,
    CITATION_VALIDATION_FAILED,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
)
from .financebench.constants import (
    DEFAULT_BM25_RRF_WEIGHT,
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_DASHSCOPE_EMBEDDING_DIM,
)
from .financebench.metrics import percentile_sorted
from .holdout.ledger_e2e import (
    account_ledger_citations,
    build_generation_prompt,
    parse_answer_payload,
    score_generated_answer,
)
from .holdout.ranking import ARM_SPECS, prepare_rerank_pool

SUITE = "ledger_structured_citation_shadow"
SUITE_VERSION = "1.0"
SCHEMA_VERSION = "lumenfin_ledger_structured_citation_shadow.v1"
CONFIG_SCHEMA_VERSION = "lumenfin_ledger_structured_citation_shadow_config.v1"
CLI_SPLIT = "public-dev"
CANONICAL_SPLIT = "public_dev"
SEAL_TAG = "ledger-public-dev-chain-v1"
SEAL_TARGET_COMMIT = "ec4d9e40d45a536ec00cbdd8fbdadf6e051e4e8c"
PROTOCOL_COMMIT = "78a719e2b744777aee5353acc24b9b88c410066e"
DEFAULT_FROZEN_CONFIG_PATH = Path("data") / "eval_rag" / "structured_citation_shadow_config.json"
DEFAULT_CACHE_MANIFEST_PATH = (
    Path("data") / "eval_rag" / "structured_citation_shadow_cache_manifest.json"
)
DEFAULT_OFFICIAL_OUTPUT_DIR = Path("outputs") / "ledger_structured_citation_shadow_v1"
LEGACY_PREFLIGHT_OUTPUT_DIR = Path("outputs") / "ledger_structured_citation_shadow_preflight_v1"
DEFAULT_PREFLIGHT_OUTPUT_DIR = Path("outputs") / "ledger_structured_citation_shadow_preflight_v2"
CACHE_MANIFEST_SCHEMA = "lumenfin_ledger_structured_citation_shadow_cache.v1"
PREVIOUS_UNUSED_CONFIG_HASH = (
    "3e834f0ed5bbd42bb8f2346968eedd0a3025f49f8db628f64b0609577c8a46ac"
)
RETIRED_BEFORE_PREFLIGHT_HASH = (
    "ef497e8b0d9ff237b21666291b98ade28a250a4e530e1e5cb57842508adb6d4e"
)
INCOMPLETE_AUDIT_CONFIG_HASH = (
    "4dd7519e13ad9eccf5a1df826fa9aa2469d5649ead8ec3c0216ee64d75f5b8ac"
)
INCOMPLETE_V1_PREFLIGHT_SHA256 = (
    "755a7f60a40e7b35f6181e210bf4a708cef5c63331d8cdf7aaf42cb3b4eefc81"
)
RETIRED_CONFIG_HASHES = {
    PREVIOUS_UNUSED_CONFIG_HASH: {
        "status": "never_executed",
        "executions": 0,
        "results": 0,
        "preflight_executions": 0,
        "accepted_preflights": 0,
        "shadow_executions": 0,
    },
    RETIRED_BEFORE_PREFLIGHT_HASH: {
        "status": "retired_before_successful_preflight",
        "executions": 0,
        "results": 0,
        "preflight_executions": 0,
        "accepted_preflights": 0,
        "shadow_executions": 0,
    },
    INCOMPLETE_AUDIT_CONFIG_HASH: {
        "status": "incomplete_preflight_audit_schema",
        "retired_reason": "incomplete_preflight_audit_schema",
        "preflight_executions": 1,
        "accepted_preflights": 0,
        "shadow_executions": 0,
        "results": 0,
        "artifact_status": "INCOMPLETE_PREFLIGHT_AUDIT_SCHEMA",
        "artifact_sha256": INCOMPLETE_V1_PREFLIGHT_SHA256,
        "accepted_for_shadow_execution": False,
        "cli_exit_code": 0,
        "shadow_results": 0,
    },
}
EVALUATION_MODE = "sealed_candidate_replay_shadow"
PREFLIGHT_SCHEMA_VERSION = "1.0"
PREFLIGHT_OK = "PREFLIGHT_OK"
PREFLIGHT_REQUIRED_FIELDS = (
    "kind",
    "preflight_schema_version",
    "status",
    "executed_at",
    "exit_code",
    "evaluation_mode",
    "cases_executed",
    "remote_request_count",
    "public_holdout_used",
    "sealed_aggregate_modified",
    "candidate_cache_modified",
)
CHAT_CREDENTIAL_KEY = "DEEPSEEK_API_KEY"
CHAIN_SEAL_RELATIVE = Path("data") / "eval_rag" / "holdout" / "ledger_public_dev_chain_seal.json"
SPLIT_MANIFEST_RELATIVE = Path("data") / "eval_rag" / "holdout" / "ledger_public_manifest.json"
SEALED_BASELINE_RELATIVE = (
    Path("data") / "eval_rag" / "holdout" / "ledger_public_dev_e2e_canary_5x10.json"
)
DEFAULT_CHAT_BASE_URL = "https://api.deepseek.com"
BILLING_SEMANTICS = "at_least_once"
HASH_SKIP_KEYS = frozenset({"config_hash"})
FORBIDDEN_SPLIT_TOKENS = frozenset(
    {
        "public_holdout",
        "public-holdout",
        "holdout",
        "confirmation",
        "test",
        "dev",
        "all",
        "public_dev",
        "public-holdout-split",
    }
)
HOLDOUT_PATH_TOKENS = ("public_holdout", "public-holdout")
REQUIRED_CREDENTIAL_KEYS = (CHAT_CREDENTIAL_KEY,)
_ACCESS_AUDIT: ContextVar["InputAccessAudit | None"] = ContextVar(
    "ledger_shadow_access_audit",
    default=None,
)
_ABS_PATH_RE = re.compile(r"(?i)([A-Z]:[\\/]|\\\\|/home/|/Users/|/tmp/|/var/)")
_HTTP_RE = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)
_SK_RE = re.compile(r"(?i)\bsk-[A-Za-z0-9_-]+")
_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|password|secret|access[_-]?key)"
)

STRUCTURED_SHADOW_SYSTEM_PROMPT = (
    "Extract one KPI number from the numbered passages. "
    "Reply with JSON only: "
    '{"answer": <string>, "citations": [<stable_chunk_id>], '
    '"structured_answer_schema_version": "1.0", '
    '"value": <number-or-null>, "abstain": <bool>}. '
    "Citations must be exact chunk_id values from the passages. "
    "Use null and abstain=true when the passages do not contain the answer. "
    "Do not invent numbers or chunk ids."
)

GenerateFn = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True)
class RuntimeSnapshot:
    model: str
    timeout_seconds: float
    max_retries: int
    retry_backoff_seconds: float
    base_url: str
    prompt: str
    rerank_provider: str
    rerank_instruct: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "base_url_sha256": chat_base_url_sha256(self.base_url),
            "prompt_sha256": prompt_sha256(self.prompt),
            "evaluation_mode": EVALUATION_MODE,
            "runtime_components": {
                "embedding": {"enabled": False, "status": "disabled"},
                "reranker": {
                    "enabled": False,
                    "status": "not_applicable",
                    "observed_env_provider": self.rerank_provider,
                },
                "chat": {"enabled": True, "provider": "deepseek", "model": self.model},
                "scoring_pool_ranker": "lexical_local",
            },
        }


class ShadowError(ValueError):
    """Fail-closed structured-citation shadow error. No secrets, no holdout text."""


@dataclass
class InputAccessAudit:
    """Records input-path access without storing secrets or holdout contents."""

    accessed_fields: list[str] = field(default_factory=list)
    holdout_path_observed: bool = False
    holdout_loader_called: bool = False
    path_guard_passed: bool = False

    def record_safe(self, field_name: str) -> None:
        self.accessed_fields.append(field_name)
        self.path_guard_passed = True

    def observe_holdout(self, field_name: str) -> None:
        self.accessed_fields.append(field_name)
        self.holdout_path_observed = True

    def mark_holdout_loader(self) -> None:
        self.holdout_loader_called = True

    def prove_holdout_unused(
        self,
        *,
        split: str,
        case_ids_in_sealed_allowlist: bool,
    ) -> dict[str, Any]:
        if split != CANONICAL_SPLIT:
            raise ShadowError("public holdout unused cannot be proven: split is not public_dev")
        if self.holdout_loader_called or self.holdout_path_observed:
            raise ShadowError("public holdout access was recorded")
        if not self.path_guard_passed:
            raise ShadowError("public holdout path guard did not pass")
        if not case_ids_in_sealed_allowlist:
            raise ShadowError("case ids are not the sealed allowlist")
        return {
            "used": False,
            "split": split,
            "path_guard_passed": True,
            "holdout_loader_called": False,
            "holdout_path_observed": False,
            "case_ids_in_sealed_allowlist": True,
            "accessed_fields": list(self.accessed_fields),
        }


def canonical_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_normalized_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def hash_material(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in HASH_SKIP_KEYS}


def compute_config_hash(payload: Mapping[str, Any]) -> str:
    return sha256_text(canonical_dumps(hash_material(payload)))


def ids_sha256(values: list[str] | tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(sorted(str(item) for item in values)).encode()).hexdigest()


def prompt_sha256(text: str | None = None) -> str:
    return sha256_text(text or STRUCTURED_SHADOW_SYSTEM_PROMPT)


def chat_base_url_sha256(url: str = DEFAULT_CHAT_BASE_URL) -> str:
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


def sha256_raw_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_prefix_rows(
    rows: list[Mapping[str, Any]],
    *,
    cases_per_company: int,
) -> list[Mapping[str, Any]]:
    blocks: list[list[Mapping[str, Any]]] = []
    current_key = None
    current: list[Mapping[str, Any]] = []
    for row in rows:
        key = str(row.get("company_key_sha256") or "")
        if current_key is None:
            current_key = key
        if key != current_key:
            blocks.append(current)
            current = []
            current_key = key
        current.append(row)
    if current:
        blocks.append(current)
    selected: list[Mapping[str, Any]] = []
    for block in blocks:
        selected.extend(block[: int(cases_per_company)])
    return selected


def chunk_ids_sha256(rows: list[Mapping[str, Any]]) -> str:
    payload = [
        {
            "query_id": str(row.get("query_id") or row.get("case_id") or ""),
            "chunk_ids": [
                str(hit.get("chunk_id") or "") for hit in (row.get("hits") or [])
            ],
        }
        for row in rows
    ]
    return sha256_text(canonical_dumps(payload))


def load_cache_rows(path: Path) -> list[dict[str, Any]]:
    safe = assert_safe_input_path(path, field="candidate-cache")
    if not safe.is_file():
        raise ShadowError("frozen candidate cache is missing")
    rows: list[dict[str, Any]] = []
    for line in safe.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ShadowError("candidate cache row is invalid")
        rows.append(payload)
    return rows


def verify_candidate_cache(
    *,
    repo_root: Path,
    config: FrozenShadowConfig,
) -> dict[str, Any]:
    rel = Path(str(config.field("candidate_cache", "manifest_path") or DEFAULT_CACHE_MANIFEST_PATH))
    manifest_path = assert_safe_input_path(repo_root / rel, field="cache-manifest")
    payload = read_json_object(manifest_path, field="cache-manifest")
    actual_hash = sha256_normalized_file(manifest_path)
    expected_hash = str(config.field("candidate_cache", "manifest_sha256") or "")
    if actual_hash != expected_hash:
        raise ShadowError("candidate cache manifest hash mismatch")
    if payload.get("schema_version") != CACHE_MANIFEST_SCHEMA:
        raise ShadowError("candidate cache manifest schema mismatch")
    if payload.get("rebuild_forbidden") is not True:
        raise ShadowError("candidate cache must forbid rebuild")
    if payload.get("embedding_fallback_forbidden") is not True:
        raise ShadowError("candidate cache must forbid embedding fallback")
    if payload.get("not_live_production_retrieval") is not True:
        raise ShadowError("candidate cache must be marked as frozen replay")
    rag = payload.get("rag_config") or {}
    expected_rag = config.field("production_rag") or {}
    for key in ("arm", "ranking_arm", "pool_strategy", "source_k", "rerank_k", "final_k"):
        if rag.get(key) != expected_rag.get(key):
            raise ShadowError("candidate cache RAG identity does not match frozen config")
    embedding = payload.get("embedding_identity") or {}
    frozen_embed = config.field("embedding") or {}
    if (
        embedding.get("provider") != frozen_embed.get("provider")
        or embedding.get("model") != frozen_embed.get("model")
        or int(embedding.get("dimension") or 0) != int(frozen_embed.get("dimension") or 0)
    ):
        raise ShadowError("candidate cache embedding identity does not match frozen config")
    rerank = payload.get("rerank_identity") or {}
    if str(rerank.get("provider") or "") != str(config.field("reranker", "provider")):
        raise ShadowError("candidate cache rerank identity does not match frozen config")
    cache_rel = Path(str(payload.get("source_path_identity") or ""))
    cache_path = assert_safe_input_path(repo_root / cache_rel, field="candidate-cache")
    report = {
        "manifest_sha256": actual_hash,
        "cache_file_sha256": str(payload.get("cache_file_sha256") or ""),
        "case_ids_sha256": str(payload.get("case_ids_sha256") or ""),
        "chunk_ids_sha256": str(payload.get("chunk_ids_sha256") or ""),
        "candidate_set_identity_sha256": str(payload.get("candidate_set_identity_sha256") or ""),
        "cache_kind": str(payload.get("cache_kind") or ""),
        "not_live_production_retrieval": True,
        "cache_present": cache_path.is_file(),
    }
    if not cache_path.is_file():
        raise ShadowError("frozen candidate cache is missing")
    before = sha256_raw_file(cache_path)
    if before != str(payload.get("cache_file_sha256") or ""):
        raise ShadowError("candidate cache file hash mismatch")
    rows = load_cache_rows(cache_path)
    after = sha256_raw_file(cache_path)
    if after != before:
        raise ShadowError("candidate cache changed during read")
    if len(rows) != int(payload.get("candidate_records_count") or -1):
        raise ShadowError("candidate cache record count mismatch")
    query_ids = [str(row.get("query_id") or "") for row in rows]
    if len(query_ids) != len(set(query_ids)):
        raise ShadowError("candidate cache contains duplicate case ids")
    parent_hash = ids_sha256(query_ids)
    if parent_hash != str(payload.get("parent_query_ids_sha256") or ""):
        raise ShadowError("candidate cache parent case id hash mismatch")
    prefix = select_prefix_rows(
        rows,
        cases_per_company=int(config.field("case_selection", "cases_per_company") or 10),
    )
    if len(prefix) != int(payload.get("case_count") or -1):
        raise ShadowError("candidate cache prefix case count mismatch")
    prefix_ids = [str(row.get("query_id") or "") for row in prefix]
    if ids_sha256(prefix_ids) != str(payload.get("case_ids_sha256") or ""):
        raise ShadowError("candidate cache prefix case id hash mismatch")
    if ids_sha256(prefix_ids) != str(config.field("case_selection", "query_ids_sha256") or ""):
        raise ShadowError("candidate cache prefix does not match frozen case selection")
    if chunk_ids_sha256(prefix) != str(payload.get("chunk_ids_sha256") or ""):
        raise ShadowError("candidate cache chunk id hash mismatch")
    hits_per = int(payload.get("hits_per_case") or 0)
    if any(len(row.get("hits") or []) != hits_per for row in prefix):
        raise ShadowError("candidate cache hit count mismatch")
    if any(
        not str(hit.get("chunk_id") or "").strip()
        for row in prefix
        for hit in (row.get("hits") or [])
    ):
        raise ShadowError("candidate cache is missing chunk ids")
    local_manifest_rel = Path(str(payload.get("source_local_manifest_identity") or ""))
    if str(local_manifest_rel):
        local_manifest = repo_root / local_manifest_rel
        if local_manifest.is_file():
            local_hash = sha256_normalized_file(
                assert_safe_input_path(local_manifest, field="local-cache-manifest")
            )
            if local_hash != str(payload.get("local_candidate_manifest_sha256") or ""):
                raise ShadowError("local candidate manifest hash mismatch")
    report["prefix_case_ids"] = prefix_ids
    report["verified"] = True
    return report


def capture_runtime_snapshot() -> RuntimeSnapshot:
    settings = LLMSettings.from_env()
    rerank_provider = (os.getenv("MAS_RAG_RERANK_PROVIDER") or "lexical").strip().casefold()
    if rerank_provider in {"dashscope", "dashscope-qwen3"}:
        rerank_provider = "qwen3"
    instruct = (
        os.getenv("MAS_RAG_RERANK_INSTRUCT") or DEFAULT_RERANK_INSTRUCT
    ).strip()
    return RuntimeSnapshot(
        model=settings.model,
        timeout_seconds=float(settings.timeout_seconds),
        max_retries=int(settings.max_retries),
        retry_backoff_seconds=float(settings.retry_backoff_seconds),
        base_url=settings.base_url,
        prompt=STRUCTURED_SHADOW_SYSTEM_PROMPT,
        rerank_provider=rerank_provider,
        rerank_instruct=instruct,
    )


def verify_snapshot_matches_frozen(
    snapshot: RuntimeSnapshot,
    config: FrozenShadowConfig,
) -> None:
    if snapshot.model != str(config.field("chat", "model")):
        raise ShadowError("runtime chat model does not match frozen config")
    if float(snapshot.timeout_seconds) != float(config.field("chat", "timeout_seconds")):
        raise ShadowError("runtime chat timeout does not match frozen config")
    if int(snapshot.max_retries) != int(config.field("chat", "max_retries")):
        raise ShadowError("runtime chat retry does not match frozen config")
    if chat_base_url_sha256(snapshot.base_url) != str(config.field("chat", "base_url_sha256")):
        raise ShadowError("runtime chat endpoint hash does not match frozen config")
    if prompt_sha256(snapshot.prompt) != str(config.field("prompts", "system_prompt_sha256")):
        raise ShadowError("runtime prompt hash does not match frozen config")
    if config.field("evaluation_mode") != EVALUATION_MODE:
        raise ShadowError("frozen config evaluation_mode is not sealed candidate replay")
    if config.field("runtime_embedding_enabled") is not False:
        raise ShadowError("sealed candidate replay forbids runtime embedding")
    if config.field("runtime_reranker_enabled") is not False:
        raise ShadowError("sealed candidate replay forbids runtime reranker")
    runtime = config.field("runtime_components") or {}
    if (runtime.get("embedding") or {}).get("enabled") is not False:
        raise ShadowError("runtime embedding must stay disabled")
    if (runtime.get("reranker") or {}).get("enabled") is not False:
        raise ShadowError("runtime reranker must stay disabled")
    if (runtime.get("chat") or {}).get("enabled") is not True:
        raise ShadowError("runtime chat must stay enabled")
    rag = config.field("production_rag") or {}
    arm = ARM_SPECS[str(rag.get("ranking_arm") or "A_prod")]
    if (
        arm.source_k != int(rag["source_k"])
        or arm.rerank_k != int(rag["rerank_k"])
        or arm.final_k != int(rag["final_k"])
        or arm.pool_strategy != str(rag["pool_strategy"])
    ):
        raise ShadowError("runtime RAG A_prod spec does not match frozen config")


def verify_runtime_matches_frozen(config: FrozenShadowConfig) -> RuntimeSnapshot:
    snapshot = capture_runtime_snapshot()
    verify_snapshot_matches_frozen(snapshot, config)
    return snapshot


def assert_exact_output_path(
    path: Path,
    expected: Path,
    *,
    repo_root: Path,
    field: str,
) -> Path:
    actual = Path(path)
    if actual.is_absolute():
        expected_resolved = (repo_root / expected).resolve()
        try:
            actual_resolved = actual.resolve()
        except OSError as exc:
            raise ShadowError(f"{field} path is invalid") from exc
        if actual_resolved != expected_resolved:
            raise ShadowError(f"{field} must be the exact frozen output directory")
        return actual
    if actual.as_posix() != expected.as_posix():
        raise ShadowError(f"{field} must be the exact frozen output directory")
    return actual


def refuse_env_remote_override() -> None:
    for key in ("LUMENFIN_SHADOW_ALLOW_REMOTE", "ALLOW_REMOTE", "MAS_ALLOW_REMOTE"):
        raw = os.getenv(key)
        if raw and raw.strip() and raw.strip().lower() not in {"0", "false", "no"}:
            raise ShadowError("environment variables cannot authorize remote shadow execution")



def git_snapshot(repo_root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        result = subprocess.run(
            args,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        return (result.stdout or "").strip()

    commit = _run(["git", "rev-parse", "HEAD"])
    porcelain = _run(["git", "status", "--porcelain"])
    return {
        "execution_commit": commit or "unknown",
        "lumenfin_commit": commit or "unknown",
        "protocol_ancestor": PROTOCOL_COMMIT,
        "worktree_dirty": bool(porcelain),
        "worktree_status": "dirty" if porcelain else "clean",
    }


def peel_seal_tag(repo_root: Path, tag: str = SEAL_TAG) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{commit}}"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    commit = (result.stdout or "").strip()
    if result.returncode != 0 or not commit:
        raise ShadowError("LEDGER seal tag is missing or cannot be peeled")
    return commit


def require_clean_worktree(repo_root: Path) -> dict[str, Any]:
    snapshot = git_snapshot(repo_root)
    if snapshot["worktree_dirty"]:
        raise ShadowError("structured citation shadow requires a clean worktree")
    return snapshot


def require_protocol_ancestor(repo_root: Path, commit: str = PROTOCOL_COMMIT) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ShadowError("HEAD is not a descendant of the frozen protocol commit")


def canonical_split(raw: str) -> str:
    token = str(raw or "").strip().casefold()
    if token != CLI_SPLIT:
        if token.replace("-", "_") in FORBIDDEN_SPLIT_TOKENS or token in FORBIDDEN_SPLIT_TOKENS:
            raise ShadowError("structured citation shadow refuses this split")
        raise ShadowError("structured citation shadow only allows --split public-dev")
    return CANONICAL_SPLIT


def _path_has_holdout_token(path: Path) -> bool:
    joined = str(path).replace("\\", "/").casefold()
    parts = [part.casefold() for part in path.parts]
    return any(token in joined or token in parts for token in HOLDOUT_PATH_TOKENS)


def assert_safe_input_path(path: str | Path, *, field: str) -> Path:
    target = Path(path)
    audit = _ACCESS_AUDIT.get()
    if _path_has_holdout_token(target):
        if audit is not None:
            audit.observe_holdout(field)
        raise ShadowError(f"{field} resolved path is forbidden")
    try:
        resolved = target if not target.exists() else target.resolve()
    except OSError as exc:
        raise ShadowError(f"{field} resolved path is forbidden") from exc
    if _path_has_holdout_token(resolved):
        if audit is not None:
            audit.observe_holdout(field)
        raise ShadowError(f"{field} resolved path is forbidden")
    if audit is not None:
        audit.record_safe(field)
    return target


def read_json_object(path: Path, *, field: str) -> dict[str, Any]:
    safe = assert_safe_input_path(path, field=field)
    if not safe.is_file():
        raise ShadowError(f"{field} is missing")
    try:
        payload = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShadowError(f"{field} is unreadable") from exc
    if not isinstance(payload, dict):
        raise ShadowError(f"{field} must be a JSON object")
    return payload


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    cleaned = sanitize_payload(payload)
    assert_safe_output(cleaned)
    atomic_write_text(
        path,
        json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n",
    )


def sanitize_error(exc: BaseException | str) -> dict[str, str]:
    if isinstance(exc, BaseException):
        category = classify_provider_exception(exc)
    else:
        category = "error"
    return {"error_type": category, "error_category": category}


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        cleaned: dict[str, Any] = {}
        for key, value in payload.items():
            lowered = str(key).casefold()
            if _SECRET_KEY_RE.search(lowered) or lowered in {"authorization", "api_key"}:
                continue
            if lowered in {"base_url", "endpoint", "url"} and isinstance(value, str):
                cleaned[f"{key}_sha256"] = sha256_text(value.strip())
                continue
            cleaned[str(key)] = sanitize_payload(value)
        return cleaned
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, str):
        text = _SK_RE.sub("[REDACTED]", payload)
        text = _HTTP_RE.sub("[REDACTED_URL]", text)
        text = _ABS_PATH_RE.sub("[REDACTED_PATH]", text)
        return text
    return payload


def assert_safe_output(payload: Any) -> None:
    blob = json.dumps(payload, ensure_ascii=False)
    if _SK_RE.search(blob):
        raise ShadowError("shadow output contains a credential-like token")
    if "Authorization" in blob or "DASHSCOPE_API_KEY=" in blob or "DEEPSEEK_API_KEY=" in blob:
        raise ShadowError("shadow output contains a credential-like token")
    if _HTTP_RE.search(blob):
        raise ShadowError("shadow output contains a raw endpoint")
    if _ABS_PATH_RE.search(blob):
        raise ShadowError("shadow output contains a local absolute path")


def published_frozen_config_fields() -> dict[str, Any]:
    a_prod = ARM_SPECS["A_prod"]
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
        "split": CANONICAL_SPLIT,
        "cli_split": CLI_SPLIT,
        "exposed_public_dev_shadow": True,
        "held_out": False,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "retuning_allowed": False,
        "public_holdout_consumed": False,
        "production_change_authorized": False,
        "financebench_phase4": "NOT_RUN",
        "ledger_seal": {
            "tag": SEAL_TAG,
            "target_commit": SEAL_TARGET_COMMIT,
            "manifest_path": CHAIN_SEAL_RELATIVE.as_posix(),
            "schema_version": "lumenfin_ledger_public_dev_chain_seal.v1",
        },
        "lumenfin_protocol_commit": PROTOCOL_COMMIT,
        "lumenfin_commit_policy": "require_clean_worktree_and_protocol_ancestor",
        "evaluation_mode": EVALUATION_MODE,
        "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        "preflight_required_fields": list(PREFLIGHT_REQUIRED_FIELDS),
        "predecessor_config": {
            "config_hash": INCOMPLETE_AUDIT_CONFIG_HASH,
            "preflight_executions": 1,
            "accepted_preflights": 0,
            "shadow_executions": 0,
            "results": 0,
            "retired_reason": "incomplete_preflight_audit_schema",
            "artifact_status": "INCOMPLETE_PREFLIGHT_AUDIT_SCHEMA",
            "artifact_sha256": INCOMPLETE_V1_PREFLIGHT_SHA256,
            "accepted_for_shadow_execution": False,
            "cli_exit_code": 0,
            "shadow_results": 0,
        },
        "not_live_production_retrieval": True,
        "candidate_cache_generation": {
            "retrieval": "hybrid_rrf_top20",
            "embedding": {
                "provider": "dashscope",
                "model": DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
                "dimension": DEFAULT_DASHSCOPE_EMBEDDING_DIM,
            },
            "reranker": {"provider": "lexical", "model": "lexical"},
            "source_commit": SEAL_TARGET_COMMIT,
            "cache_file_sha256": (
                "c49d06665376b769950492cecd41cb3d7ad144509e57d0cdf09493aeab52e65a"
            ),
            "manifest_sha256": (
                "2550d0310caaa68f13107e8c0f870d891bda3797908b5a888e30b49048b9db90"
            ),
        },
        "runtime_components": {
            "embedding": {"enabled": False, "status": "disabled"},
            "reranker": {"enabled": False, "status": "not_applicable"},
            "chat": {"enabled": True, "provider": "deepseek"},
            "scoring_pool_ranker": "lexical_local",
        },
        "runtime_embedding_enabled": False,
        "runtime_reranker_enabled": False,
        "candidate_cache": {
            "manifest_path": DEFAULT_CACHE_MANIFEST_PATH.as_posix(),
            "manifest_sha256": "2550d0310caaa68f13107e8c0f870d891bda3797908b5a888e30b49048b9db90",
            "cache_kind": "frozen_hybrid_candidate_replay",
            "not_live_production_retrieval": True,
        },
        "split_manifest": {
            "path": SPLIT_MANIFEST_RELATIVE.as_posix(),
            "sha256": "889329d3647a3c4543d12fa0ebae0172969e787bb877f73c2f0aee967a4920d5",
            "schema_version": "lumenfin_public_benchmark_manifest.v1",
            "public_dev_query_ids_sha256": (
                "b0e2163c2d9d0e4d5a948313bde2507418b552e22738d6f4c2f7825766dca65c"
            ),
        },
        "case_selection": {
            "source_artifact": SEALED_BASELINE_RELATIVE.as_posix(),
            "strategy": "frozen_5x50_company_prefix_v1",
            "companies": 5,
            "cases_per_company": 10,
            "query_count": 50,
            "query_ids_sha256": "6fbe540fa4cca45f298950b7728d769beee8bb43a9711c3bece01a2b62a8f9aa",
            "parent_query_ids_sha256": (
                "cb1654a41dec7ae04efd6666dd5ddfbcf29862631b1d0acbad884fb0402de044"
            ),
        },
        "sealed_baseline": {
            "path": SEALED_BASELINE_RELATIVE.as_posix(),
            "sha256": "a7b4868da2694118e31aad62a840453f5acf44dbb5a8e7593b0e3197db3d2b21",
            "schema_version": "lumenfin_ledger_e2e_canary.v1",
            "e2e_source_sha256": "382a81525103276508ff25b269e699e061f952621e6df607d187a0528f7395a4",
            "readonly": True,
            "comparison_arm": "lexical",
        },
        "dataset": {
            "dataset_id": "artefactory/ledger-long-context-KPI-QA",
            "dataset_revision": "b7085dc6cb16b3ec8149a9baf6dd2d3416cf7619",
            "dataset_snapshot_sha256": (
                "6449593accbf71ca282f28b424b6e5d267dd7f180ccd18238614d92b412d44ea"
            ),
            "source_artifact_sha256": (
                "405eb7c805db90258e4246651688b8d8bef89c77d4a4ce2cbcbf9e5fa4bfe9ad"
            ),
        },
        "chat": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "timeout_seconds": 45.0,
            "max_retries": 3,
            "retry_backoff_seconds": 0.5,
            "temperature": 0.0,
            "max_tokens": 200,
            "base_url_sha256": chat_base_url_sha256(),
        },
        "embedding": {
            "provider": "dashscope",
            "model": DEFAULT_DASHSCOPE_EMBEDDING_MODEL,
            "dimension": DEFAULT_DASHSCOPE_EMBEDDING_DIM,
            "used_in_this_suite": False,
            "reason": "frozen_hybrid_candidate_cache_no_reembed",
        },
        "reranker": {
            "provider": "lexical",
            "model": "lexical",
            "used_in_runtime": False,
            "instruct": DEFAULT_RERANK_INSTRUCT,
            "timeout_seconds": 12.0,
            "max_attempts": 2,
            "backoff_seconds": 0.25,
            "max_inflight": 2,
            "max_document_chars": 4000,
        },
        "production_rag": {
            "arm": "A",
            "ranking_arm": "A_prod",
            "pool_strategy": a_prod.pool_strategy,
            "source_k": a_prod.source_k,
            "rerank_k": a_prod.rerank_k,
            "final_k": a_prod.final_k,
            "bm25_rrf_weight": DEFAULT_BM25_RRF_WEIGHT,
            "rerank_enabled": True,
        },
        "prompts": {
            "version": "ledger_structured_citation_shadow_v1",
            "system_prompt_sha256": prompt_sha256(),
        },
        "timeouts": {
            "chat_timeout_seconds": 45.0,
            "rerank_timeout_seconds": 12.0,
        },
        "retry": {
            "chat_max_retries": 3,
            "rerank_max_attempts": 2,
        },
        "concurrency": {
            "workers": 1,
            "max_inflight": 1,
        },
        "bootstrap": {
            "seed": DEFAULT_BOOTSTRAP_SEED,
            "samples": DEFAULT_BOOTSTRAP_SAMPLES,
        },
        "output": {
            "schema_version": SCHEMA_VERSION,
            "official_dirname": DEFAULT_OFFICIAL_OUTPUT_DIR.name,
            "preflight_dirname": DEFAULT_PREFLIGHT_OUTPUT_DIR.name,
            "legacy_preflight_dirname": LEGACY_PREFLIGHT_OUTPUT_DIR.name,
            "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
        },
        "call_budget": {
            "cases_total": 50,
            "generate_logical_calls_expected": 50,
            "rerank_remote_calls_expected": 0,
            "embedding_remote_calls_expected": 0,
            "remote_calls_expected": 50,
            "billing_semantics": BILLING_SEMANTICS,
            "exactly_once": False,
        },
        "evaluation_metrics": [
            "cases_total",
            "cases_succeeded",
            "cases_failed",
            "provider_errors",
            "remote_request_count",
            "latency_p50",
            "latency_p95",
            "structured_answer_present",
            "structured_emission_rate",
            "citation_source_distribution",
            "answers_with_citations",
            "citations_total",
            "valid_citations",
            "unknown_citations",
            "unverified_citations",
            "cross_scope_citations",
            "stale_citations",
            "citation_validation_failed",
            "claims_total",
            "supported_claims",
            "unsupported_claims",
            "citation_support_rate",
            "answers_fully_supported",
            "answers_partially_supported",
            "answers_unsupported",
            "complete",
            "incomplete_data",
            "degraded",
            "failed",
        ],
        "paired_comparison": {
            "kind": "post-hoc exposed comparison",
            "held_out": False,
            "suitable_for_model_selection": False,
            "writes_sealed_aggregate": False,
        },
        "config_hash_canonicalization": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "excluded_keys": sorted(HASH_SKIP_KEYS),
        },
    }


def published_config_hash() -> str:
    return compute_config_hash(published_frozen_config_fields())


class FrozenShadowConfig:
    def __init__(self, path: Path, payload: dict[str, Any]) -> None:
        self.path = path
        self.payload = payload
        self.config_hash = str(payload["config_hash"])

    def field(self, *keys: str, default: Any = None) -> Any:
        cursor: Any = self.payload
        for key in keys:
            if not isinstance(cursor, Mapping) or key not in cursor:
                return default
            cursor = cursor[key]
        return cursor


def load_frozen_config(
    path: str | Path,
    *,
    require_published: bool = False,
) -> FrozenShadowConfig:
    config_path = assert_safe_input_path(path, field="frozen-config")
    payload = read_json_object(config_path, field="frozen-config")
    stored = str(payload.get("config_hash") or "").strip()
    computed = compute_config_hash(payload)
    if not stored:
        raise ShadowError("frozen config is missing config_hash")
    if stored != computed:
        raise ShadowError("frozen config_hash does not match canonical digest")
    if stored in RETIRED_CONFIG_HASHES:
        raise ShadowError("retired shadow config hash is not executable")
    if require_published and stored != published_config_hash():
        raise ShadowError("frozen config_hash is not the published shadow digest")
    _validate_frozen_payload(payload)
    return FrozenShadowConfig(config_path, payload)


def _validate_frozen_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ShadowError("frozen config schema_version is unsupported")
    if payload.get("suite") != SUITE:
        raise ShadowError("frozen config suite mismatch")
    if payload.get("split") != CANONICAL_SPLIT:
        raise ShadowError("frozen config split mismatch")
    if payload.get("structured_answer_schema_version") != STRUCTURED_ANSWER_SCHEMA_VERSION:
        raise ShadowError("frozen config structured_answer_schema_version mismatch")
    if payload.get("held_out") is not False:
        raise ShadowError("frozen config must set held_out false")
    if payload.get("product_accuracy_claim") is not False:
        raise ShadowError("frozen config must not claim product accuracy")
    if payload.get("benchmark_claim") is not False:
        raise ShadowError("frozen config must not claim a benchmark")
    if payload.get("retuning_allowed") is not False:
        raise ShadowError("frozen config must forbid retuning")
    if payload.get("evaluation_mode") != EVALUATION_MODE:
        raise ShadowError("frozen config must declare sealed candidate replay")
    if payload.get("runtime_embedding_enabled") is not False:
        raise ShadowError("frozen config must disable runtime embedding")
    if payload.get("runtime_reranker_enabled") is not False:
        raise ShadowError("frozen config must disable runtime reranker")
    if payload.get("preflight_schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ShadowError("frozen config preflight_schema_version mismatch")
    if list(payload.get("preflight_required_fields") or []) != list(PREFLIGHT_REQUIRED_FIELDS):
        raise ShadowError("frozen config preflight required fields mismatch")
    output = payload.get("output") or {}
    if output.get("preflight_dirname") != DEFAULT_PREFLIGHT_OUTPUT_DIR.name:
        raise ShadowError("frozen config preflight directory must be v2")
    if output.get("official_dirname") != DEFAULT_OFFICIAL_OUTPUT_DIR.name:
        raise ShadowError("frozen config official directory mismatch")
    if payload.get("config_hash") in RETIRED_CONFIG_HASHES:
        raise ShadowError("retired shadow config hash is not executable")
    cache = payload.get("candidate_cache") or {}
    if cache.get("not_live_production_retrieval") is not True:
        raise ShadowError("frozen config must mark candidate cache as frozen replay")
    if not str(cache.get("manifest_path") or "").strip():
        raise ShadowError("frozen config is missing candidate cache manifest path")
    if len(str(cache.get("manifest_sha256") or "")) != 64:
        raise ShadowError("frozen config is missing candidate cache manifest hash")
    seal = payload.get("ledger_seal") or {}
    if seal.get("tag") != SEAL_TAG or seal.get("target_commit") != SEAL_TARGET_COMMIT:
        raise ShadowError("frozen config LEDGER seal identity mismatch")
    blob = json.dumps(payload, ensure_ascii=False)
    if _SK_RE.search(blob) or _SECRET_KEY_RE.search(blob):
        raise ShadowError("frozen config contains a credential-like field")
    if _HTTP_RE.search(blob):
        raise ShadowError("frozen config contains a raw endpoint")
    if _ABS_PATH_RE.search(blob):
        raise ShadowError("frozen config contains a local absolute path")


def credential_presence(*, repo_root: Path) -> list[dict[str, Any]]:
    reports = []
    for item in describe_credential_sources(root=repo_root, keys=REQUIRED_CREDENTIAL_KEYS):
        reports.append(
            {
                "key": item.key,
                "source": item.source,
                "present": item.source != "unset",
            }
        )
    return reports


def require_chat_credential(*, repo_root: Path) -> dict[str, Any]:
    reports = credential_presence(repo_root=repo_root)
    chat = next((item for item in reports if item["key"] == CHAT_CREDENTIAL_KEY), None)
    if chat is None or not chat.get("present"):
        raise ShadowError("formal shadow requires DEEPSEEK_API_KEY")
    return {"key": CHAT_CREDENTIAL_KEY, "source": chat["source"], "present": True}


def forbid_runtime_embedding_call() -> None:
    raise ShadowError("sealed candidate replay forbids embedding calls")


def forbid_runtime_reranker_call() -> None:
    raise ShadowError("sealed candidate replay forbids runtime reranker calls")


def bind_chain_seal(
    *,
    repo_root: Path,
    config: FrozenShadowConfig,
    verify_tag: bool = True,
) -> dict[str, Any]:
    seal_rel = Path(str(config.field("ledger_seal", "manifest_path")))
    seal_path = assert_safe_input_path(repo_root / seal_rel, field="chain-seal")
    seal = read_json_object(seal_path, field="chain-seal")
    if seal.get("split") != CANONICAL_SPLIT:
        raise ShadowError("chain seal split is not public_dev")
    if seal.get("public_holdout_consumed") is not False:
        raise ShadowError("chain seal reports public_holdout consumed")
    if str(seal.get("recommended_annotated_tag") or "") != SEAL_TAG:
        raise ShadowError("chain seal tag mismatch")
    if str(seal.get("recommended_tag_target_commit") or "") != SEAL_TARGET_COMMIT:
        raise ShadowError("chain seal target commit mismatch")
    if verify_tag:
        peeled = peel_seal_tag(repo_root, SEAL_TAG)
        if peeled != SEAL_TARGET_COMMIT:
            raise ShadowError("local seal tag peeled commit mismatch")
    manifest_rel = Path(str(config.field("split_manifest", "path")))
    manifest_path = assert_safe_input_path(repo_root / manifest_rel, field="split-manifest")
    actual_manifest_hash = sha256_normalized_file(manifest_path)
    expected_manifest_hash = str(config.field("split_manifest", "sha256"))
    if actual_manifest_hash != expected_manifest_hash:
        raise ShadowError("public/dev split manifest hash mismatch")
    baseline_rel = Path(str(config.field("sealed_baseline", "path")))
    baseline_path = assert_safe_input_path(repo_root / baseline_rel, field="sealed-baseline")
    actual_baseline_hash = sha256_normalized_file(baseline_path)
    expected_baseline_hash = str(config.field("sealed_baseline", "sha256"))
    if actual_baseline_hash != expected_baseline_hash:
        raise ShadowError("sealed baseline hash mismatch")
    baseline = read_json_object(baseline_path, field="sealed-baseline")
    selection = baseline.get("selection") or {}
    if str(selection.get("query_ids_sha256") or "") != str(
        config.field("case_selection", "query_ids_sha256")
    ):
        raise ShadowError("sealed case id hash mismatch")
    if int(baseline.get("cases") or 0) != int(config.field("case_selection", "query_count")):
        raise ShadowError("sealed case count mismatch")
    return {
        "seal_tag": SEAL_TAG,
        "seal_commit": SEAL_TARGET_COMMIT,
        "split_manifest_sha256": actual_manifest_hash,
        "sealed_baseline_sha256": actual_baseline_hash,
        "query_ids_sha256": str(selection.get("query_ids_sha256") or ""),
        "baseline_readonly": True,
    }


def read_sealed_baseline_readonly(
    *,
    repo_root: Path,
    config: FrozenShadowConfig,
) -> dict[str, Any]:
    path = assert_safe_input_path(
        repo_root / Path(str(config.field("sealed_baseline", "path"))),
        field="sealed-baseline",
    )
    before = sha256_normalized_file(path)
    payload = read_json_object(path, field="sealed-baseline")
    after = sha256_normalized_file(path)
    if before != after or before != str(config.field("sealed_baseline", "sha256")):
        raise ShadowError("sealed baseline changed during read")
    comparison = payload.get("comparison") or {}
    arm = comparison.get(str(config.field("sealed_baseline", "comparison_arm") or "lexical")) or {}
    cases = int(payload.get("cases") or 0)
    support_rate = float(arm.get("citation_support_rate") or 0.0)
    abstain_rate = float(arm.get("abstain_rate") or 0.0)
    return {
        "cases": cases,
        "structured_answer_present": 0,
        "valid_citations": 0,
        "supported_claims": int(round(support_rate * cases)),
        "citation_support_rate": support_rate,
        "incomplete_or_degraded": int(round(abstain_rate * cases)),
        "citation_source_distribution": {CITATION_SOURCE_UNAVAILABLE: cases},
        "readonly": True,
        "product_accuracy_claim": False,
    }


def assert_case_ids(
    actual: list[str],
    *,
    expected_ids: list[str] | None = None,
    expected_hash: str | None = None,
) -> None:
    if not actual:
        raise ShadowError("shadow case selection is empty")
    if len(actual) != len(set(actual)):
        raise ShadowError("shadow case selection contains duplicate ids")
    if expected_ids is not None:
        if actual != list(expected_ids):
            extra = [item for item in actual if item not in expected_ids]
            missing = [item for item in expected_ids if item not in actual]
            if extra or missing:
                raise ShadowError("shadow case id set does not match the public/dev seal allowlist")
            raise ShadowError("shadow case id order does not match the public/dev seal allowlist")
    if expected_hash and ids_sha256(actual) != expected_hash:
        raise ShadowError("shadow case id hash does not match the public/dev seal")


def load_case_fixture(path: str | Path, *, allowlist: list[str], expected_hash: str) -> list[dict[str, Any]]:
    safe = assert_safe_input_path(path, field="cases")
    payload = json.loads(safe.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("cases")
    else:
        rows = payload
    if not isinstance(rows, list) or any(not isinstance(item, dict) for item in rows):
        raise ShadowError("cases fixture is invalid")
    case_ids = [str(item.get("case_id") or item.get("query_id") or "") for item in rows]
    if any(not item for item in case_ids):
        raise ShadowError("cases fixture is missing case ids")
    assert_case_ids(case_ids, expected_ids=allowlist, expected_hash=expected_hash)
    normalized: list[dict[str, Any]] = []
    for item, case_id in zip(rows, case_ids):
        hits = list(item.get("hits") or [])
        qrels = item.get("qrels") or {}
        if isinstance(qrels, list):
            qrels = {str(row.get("doc_id") or ""): int(row.get("relevance") or 0) for row in qrels}
        normalized.append(
            {
                "case_id": case_id,
                "query_text": str(item.get("query_text") or ""),
                "gold_value": float(item.get("gold_value") or 0.0),
                "hits": hits,
                "qrels": {str(key): int(value) for key, value in dict(qrels).items()},
                "tenant_id": str(item.get("tenant_id") or "default"),
                "session_id": str(item.get("session_id") or "shadow"),
            }
        )
    return normalized


def rank_hits(hits: list[dict[str, Any]], *, query_text: str, final_k: int) -> list[dict[str, Any]]:
    pool = prepare_rerank_pool(hits, arm="A_prod")
    ranker = LexicalReranker()
    ranked, meta = ranker.rerank(query_text, pool, top_k=final_k)
    if str(meta.get("rerank_provider") or "") != "lexical":
        raise ShadowError("scoring pool ranker must stay local lexical")
    if int(meta.get("rerank_tokens") or 0) or str(meta.get("rerank_error_type") or ""):
        raise ShadowError("scoring pool ranker performed a remote call")
    return ranked


def classify_outcome(row: Mapping[str, Any]) -> str:
    if row.get("failed"):
        return "failed"
    accounting = row.get("citation_accounting") or {}
    if row.get("citation_validation_failed") or accounting.get("cross_run_or_tenant_citation"):
        return "degraded"
    if row.get("abstain") or not row.get("citations"):
        return "incomplete_data"
    if row.get("structured_answer_present") and row.get("supported_claim"):
        return "complete"
    if row.get("structured_answer_present"):
        return "degraded"
    return "incomplete_data"


def score_case(
    case: Mapping[str, Any],
    *,
    raw: str,
    latency_ms: float,
    generate_attempts: int,
    remote_calls: int,
) -> dict[str, Any]:
    parsed = parse_answer_payload(raw)
    ranked = rank_hits(
        list(case["hits"]),
        query_text=str(case.get("query_text") or ""),
        final_k=ARM_SPECS["A_prod"].final_k,
    )
    scored = score_generated_answer(
        gold_value=float(case["gold_value"]),
        parsed={
            **parsed,
            "tenant_id": case.get("tenant_id") or "",
            "session_id": case.get("session_id") or "",
        },
        hits=ranked,
        qrels=case["qrels"],
    )
    accounting = scored["citation_accounting"]
    citations = list(parsed.get("citations") or [])
    source = str(scored.get("citation_source") or CITATION_SOURCE_UNAVAILABLE)
    structured_present = (
        source == CITATION_SOURCE_STRUCTURED and bool(citations) and bool(parsed.get("structured_answer_schema_version"))
    )
    validation_failed = False
    if citations and int(accounting.get("valid_citation") or 0) == 0:
        validation_failed = True
    stale = int(accounting.get("unverified_citation") or 0)
    row = {
        "case_id": case["case_id"],
        "complete": True,
        "failed": False,
        "abstain": bool(scored.get("abstain")),
        "structured_answer_present": structured_present,
        "citation_source": source,
        "citations": citations,
        "citations_total": len(citations),
        "valid_citations": int(accounting.get("valid_citation") or 0),
        "unknown_citations": int(accounting.get("unknown_citation") or 0),
        "unverified_citations": int(accounting.get("unverified_citation") or 0),
        "cross_scope_citations": int(accounting.get("cross_run_or_tenant_citation") or 0),
        "stale_citations": stale,
        "citation_validation_failed": validation_failed,
        "supported_claim": bool(scored.get("citation_supported")),
        "unsupported_claim": bool(citations) and not bool(scored.get("citation_supported")),
        "claims_total": 1 if citations or parsed.get("answer") else 0,
        "citation_accounting": accounting,
        "latency_ms": round(float(latency_ms), 2),
        "generate_attempts": int(generate_attempts),
        "remote_calls": int(remote_calls),
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "product_accuracy_claim": False,
        "held_out": False,
        "citation_path": (
            CITATION_PATH_VALIDATION_FAILED if validation_failed else source
        ),
        "citation_validation": (
            CITATION_VALIDATION_FAILED if validation_failed else "passed"
        ),
    }
    if row["claims_total"] == 0:
        row["claims_total"] = 1 if parsed.get("answer") or citations else 0
    if row["supported_claim"]:
        row["claim_support"] = "supported"
    elif citations:
        row["claim_support"] = "unsupported"
    else:
        row["claim_support"] = "unavailable"
    row["outcome"] = classify_outcome(row)
    return row


def failed_case(case: Mapping[str, Any], exc: BaseException, *, remote_calls: int) -> dict[str, Any]:
    redacted = sanitize_error(exc)
    return {
        "case_id": case["case_id"],
        "complete": True,
        "failed": True,
        "outcome": "failed",
        "provider_error": True,
        "structured_answer_present": False,
        "citation_source": CITATION_SOURCE_UNAVAILABLE,
        "citations": [],
        "citations_total": 0,
        "valid_citations": 0,
        "unknown_citations": 0,
        "unverified_citations": 0,
        "cross_scope_citations": 0,
        "stale_citations": 0,
        "citation_validation_failed": False,
        "supported_claim": False,
        "unsupported_claim": False,
        "claims_total": 0,
        "claim_support": "unavailable",
        "latency_ms": 0.0,
        "generate_attempts": 1,
        "remote_calls": int(remote_calls),
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "product_accuracy_claim": False,
        "held_out": False,
        **redacted,
    }


def summarize_rows(
    rows: list[Mapping[str, Any]],
    *,
    cases_total: int,
    remote_request_count: int,
    sealed: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    succeeded = [row for row in rows if not row.get("failed")]
    failed = [row for row in rows if row.get("failed")]
    latencies = sorted(float(row.get("latency_ms") or 0.0) for row in succeeded)
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("citation_source") or CITATION_SOURCE_UNAVAILABLE)
        source_counts[source] = source_counts.get(source, 0) + 1
    structured = sum(1 for row in rows if row.get("structured_answer_present"))
    with_citations = sum(1 for row in rows if row.get("citations"))
    claims_total = sum(int(row.get("claims_total") or 0) for row in rows)
    supported = sum(1 for row in rows if row.get("supported_claim"))
    unsupported = sum(1 for row in rows if row.get("unsupported_claim"))
    fully = sum(
        1
        for row in rows
        if row.get("supported_claim") and not row.get("unsupported_claim") and row.get("citations")
    )
    partial = 0
    answers_unsupported = sum(
        1 for row in rows if row.get("citations") and not row.get("supported_claim")
    )
    outcomes = {name: 0 for name in ("complete", "incomplete_data", "degraded", "failed")}
    for row in rows:
        outcomes[str(row.get("outcome") or "failed")] = (
            outcomes.get(str(row.get("outcome") or "failed"), 0) + 1
        )
    valid = sum(int(row.get("valid_citations") or 0) for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "suite_version": SUITE_VERSION,
        "exposed_public_dev_shadow": True,
        "held_out": False,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "retuning_allowed": False,
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "cases_total": cases_total,
        "cases_succeeded": len(succeeded),
        "cases_failed": len(failed),
        "provider_errors": len(failed),
        "remote_request_count": int(remote_request_count),
        "latency_p50": round(percentile_sorted(latencies, 0.5), 2) if latencies else 0.0,
        "latency_p95": round(percentile_sorted(latencies, 0.95), 2) if latencies else 0.0,
        "structured_answer_present": structured,
        "structured_emission_rate": round(structured / cases_total, 4) if cases_total else 0.0,
        "citation_source_distribution": source_counts,
        "answers_with_citations": with_citations,
        "citations_total": sum(int(row.get("citations_total") or 0) for row in rows),
        "valid_citations": valid,
        "unknown_citations": sum(int(row.get("unknown_citations") or 0) for row in rows),
        "unverified_citations": sum(int(row.get("unverified_citations") or 0) for row in rows),
        "cross_scope_citations": sum(int(row.get("cross_scope_citations") or 0) for row in rows),
        "stale_citations": sum(int(row.get("stale_citations") or 0) for row in rows),
        "citation_validation_failed": sum(
            1 for row in rows if row.get("citation_validation_failed")
        ),
        "claims_total": claims_total,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "citation_support_rate": round(supported / cases_total, 4) if cases_total else 0.0,
        "answers_fully_supported": fully,
        "answers_partially_supported": partial,
        "answers_unsupported": answers_unsupported,
        **outcomes,
    }
    if sealed is not None:
        summary["paired_vs_sealed"] = paired_descriptive_comparison(summary, sealed)
    return summary


def paired_descriptive_comparison(
    shadow: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> dict[str, Any]:
    def _delta(name: str, sealed_name: str | None = None) -> int:
        return int(shadow.get(name) or 0) - int(sealed.get(sealed_name or name) or 0)

    return {
        "comparison_kind": "post-hoc exposed comparison",
        "held_out": False,
        "not_held_out": True,
        "suitable_for_model_selection": False,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "writes_sealed_aggregate": False,
        "structured_present_delta": _delta("structured_answer_present"),
        "valid_citation_delta": _delta("valid_citations", "valid_citations"),
        "supported_claim_delta": _delta("supported_claims"),
        "incomplete_or_degraded_delta": (
            int(shadow.get("incomplete_data") or 0)
            + int(shadow.get("degraded") or 0)
            - int(sealed.get("incomplete_or_degraded") or 0)
        ),
    }


def require_fresh_or_resume(path: Path, *, resume: bool) -> None:
    if not path.exists():
        return
    if any(path.iterdir()) and not resume:
        raise ShadowError("refusing to overwrite a non-empty shadow output directory")


def _read_complete_per_case(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for index, line in enumerate(raw_lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            if index == len(raw_lines) - 1:
                continue
            raise ShadowError("per_case.jsonl is corrupt; refusing resume") from None
        if not isinstance(payload, dict):
            raise ShadowError("per_case.jsonl is corrupt; refusing resume")
        case_id = str(payload.get("case_id") or "").strip()
        if case_id and payload.get("complete") is True:
            rows.append(payload)
    ids = [str(row["case_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ShadowError("per_case.jsonl contains duplicate case ids")
    return rows


def _resume_ids_from(payload: Mapping[str, Any], *, label: str) -> set[str]:
    raw = payload.get("completed_case_ids")
    if not isinstance(raw, list) or any(not str(item).strip() for item in raw):
        raise ShadowError(f"{label} is missing completed case IDs")
    ids = {str(item) for item in raw}
    completed = payload.get("completed_cases")
    if completed is None:
        raise ShadowError(f"{label} is missing completed_cases")
    try:
        count = int(completed)
    except (TypeError, ValueError) as exc:
        raise ShadowError(f"{label} completed_cases is invalid") from exc
    if count != len(ids):
        raise ShadowError(f"{label} completed_cases does not match completed case IDs")
    return ids


def _require_resume_nesting(
    *,
    manifest_ids: set[str],
    checkpoint_ids: set[str],
    per_case_ids: set[str],
) -> None:
    extra = (manifest_ids | checkpoint_ids) - per_case_ids
    if extra:
        raise ShadowError("resume metadata claims cases missing from per_case.jsonl")
    pairs = (
        (manifest_ids, checkpoint_ids),
        (manifest_ids, per_case_ids),
        (checkpoint_ids, per_case_ids),
    )
    if any(not (left <= right or right <= left) for left, right in pairs):
        raise ShadowError("completed case IDs diverge across manifest/checkpoint/per_case")


def validate_resume_identity(
    *,
    output_dir: Path,
    git_commit: str,
    config_hash: str,
    seal_hash: str,
    case_manifest_hash: str,
    cache_hash: str,
    provider: str,
    model: str,
    protocol_ancestor: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    manifest = read_json_object(output_dir / "manifest.json", field="manifest")
    checkpoint = read_json_object(output_dir / "checkpoint.json", field="checkpoint")
    rows = _read_complete_per_case(output_dir / "per_case.jsonl")
    per_case_ids = {str(row["case_id"]) for row in rows}
    expected = {
        "execution_commit": git_commit,
        "protocol_ancestor": protocol_ancestor,
        "config_hash": config_hash,
        "seal_hash": seal_hash,
        "case_manifest_hash": case_manifest_hash,
        "candidate_cache_hash": cache_hash,
        "provider": provider,
        "model": model,
        "output_dirname": output_dir.name,
        "billing_semantics": BILLING_SEMANTICS,
    }
    for label, payload in (("manifest.json", manifest), ("checkpoint.json", checkpoint)):
        for key, value in expected.items():
            if str(payload.get(key) or "") != str(value):
                raise ShadowError(f"refusing resume: {label} {key} mismatch")
        if payload.get("exactly_once") is True:
            raise ShadowError("refusing resume: output claims exactly-once")
    _require_resume_nesting(
        manifest_ids=_resume_ids_from(manifest, label="manifest.json"),
        checkpoint_ids=_resume_ids_from(checkpoint, label="checkpoint.json"),
        per_case_ids=per_case_ids,
    )
    return rows, manifest, checkpoint


def identity_payload(
    *,
    execution_commit: str,
    protocol_ancestor: str,
    config_hash: str,
    seal_hash: str,
    case_manifest_hash: str,
    cache_hash: str,
    provider: str,
    model: str,
    prompt_sha256_value: str,
    output_dir: Path,
    completed_ids: list[str],
    cases_total: int,
    calls_expected: int,
    calls_total: int,
    calls_this_invocation: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE,
        "protocol_ancestor": protocol_ancestor,
        "execution_commit": execution_commit,
        "lumenfin_commit": execution_commit,
        "config_hash": config_hash,
        "seal_hash": seal_hash,
        "case_manifest_hash": case_manifest_hash,
        "candidate_cache_hash": cache_hash,
        "provider": provider,
        "model": model,
        "prompt_sha256": prompt_sha256_value,
        "output_dirname": output_dir.name,
        "completed_case_ids": completed_ids,
        "completed_cases": len(completed_ids),
        "cases_total": cases_total,
        "cases_remaining": max(0, cases_total - len(completed_ids)),
        "calls_expected": calls_expected,
        "calls_total": calls_total,
        "calls_this_invocation": calls_this_invocation,
        "billing_semantics": BILLING_SEMANTICS,
        "exactly_once": False,
        "unobserved_inflight_remote_calls_possible": True,
        "held_out": False,
        "product_accuracy_claim": False,
        "benchmark_claim": False,
        "exposed_public_dev_shadow": True,
        "not_live_production_retrieval": True,
        "evaluation_mode": EVALUATION_MODE,
    }


def append_completed_case(path: Path, row: Mapping[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    cleaned = sanitize_payload(dict(row))
    assert_safe_output(cleaned)
    atomic_write_text(path, existing + json.dumps(cleaned, ensure_ascii=False) + "\n")


class NetworkProbe:
    """Count and block outbound socket connects."""

    def __init__(self) -> None:
        self.remote_request_count = 0
        self._installed = False
        self._orig_connect: Callable[..., Any] | None = None
        self._orig_create: Callable[..., Any] | None = None

    def install(self) -> None:
        if self._installed:
            return
        self._orig_connect = socket.socket.connect
        self._orig_create = socket.create_connection
        probe = self

        def connect(sock: socket.socket, *args: Any, **kwargs: Any) -> Any:
            probe.remote_request_count += 1
            raise OSError("structured citation shadow forbids network")

        def block(*_args: Any, **_kwargs: Any) -> Any:
            probe.remote_request_count += 1
            raise OSError("structured citation shadow forbids network")

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.create_connection = block  # type: ignore[assignment]
        self._installed = True

    def remove(self) -> None:
        if not self._installed:
            return
        if self._orig_connect is not None:
            socket.socket.connect = self._orig_connect  # type: ignore[method-assign]
        if self._orig_create is not None:
            socket.create_connection = self._orig_create  # type: ignore[assignment]
        self._installed = False

    def __enter__(self) -> "NetworkProbe":
        self.install()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.remove()


def render_results_md(summary: Mapping[str, Any], *, preflight: bool = False) -> str:
    kind = "preflight" if preflight else "shadow summary"
    lines = [
        f"# LEDGER structured citation {kind}",
        "",
        "This is an exposed public/dev shadow. It is not held-out, not product",
        "accuracy, not a LEDGER benchmark, and not an rc5 score.",
        "",
        f"- cases_total: {summary.get('cases_total', 0)}",
        f"- remote_request_count: {summary.get('remote_request_count', 0)}",
        f"- billing_semantics: {BILLING_SEMANTICS}",
        f"- exactly_once: false",
        "",
        "A crash after a successful remote call but before the atomic",
        "`per_case.jsonl` write may bill twice on resume.",
        "",
    ]
    return "\n".join(lines)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _readonly_artifact_hashes(
    *,
    repo_root: Path,
    config: FrozenShadowConfig,
) -> dict[str, str]:
    manifest_rel = Path(str(config.field("candidate_cache", "manifest_path") or DEFAULT_CACHE_MANIFEST_PATH))
    manifest_path = assert_safe_input_path(repo_root / manifest_rel, field="cache-manifest")
    if not manifest_path.is_file():
        raise ShadowError("candidate cache manifest is missing")
    manifest = read_json_object(manifest_path, field="cache-manifest")
    cache_rel = Path(str(manifest.get("source_path_identity") or ""))
    cache_path = assert_safe_input_path(repo_root / cache_rel, field="candidate-cache")
    if not cache_path.is_file():
        raise ShadowError("frozen candidate cache is missing")
    baseline_rel = Path(str(config.field("sealed_baseline", "path")))
    baseline_path = assert_safe_input_path(repo_root / baseline_rel, field="sealed-baseline")
    if not baseline_path.is_file():
        raise ShadowError("sealed baseline is missing")
    return {
        "manifest_sha256": sha256_normalized_file(manifest_path),
        "cache_file_sha256": sha256_raw_file(cache_path),
        "sealed_baseline_sha256": sha256_normalized_file(baseline_path),
    }


def _assert_preflight_success_contract(report: Mapping[str, Any]) -> None:
    missing = [key for key in PREFLIGHT_REQUIRED_FIELDS if key not in report]
    if missing:
        raise ShadowError("preflight report is missing required audit fields")
    if report.get("kind") != "preflight":
        raise ShadowError("preflight report kind is invalid")
    if report.get("preflight_schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ShadowError("preflight schema version is invalid")
    if report.get("status") != PREFLIGHT_OK:
        raise ShadowError("preflight status is not PREFLIGHT_OK")
    if report.get("exit_code") != 0:
        raise ShadowError("preflight success report must set exit_code 0")
    if report.get("cases_executed") != 0 or report.get("remote_request_count") != 0:
        raise ShadowError("preflight success cannot record cases or remote calls")
    if report.get("public_holdout_used") is not False:
        raise ShadowError("preflight success cannot record public_holdout use")
    if report.get("sealed_aggregate_modified") is not False:
        raise ShadowError("preflight success cannot record sealed aggregate mutation")
    if report.get("candidate_cache_modified") is not False:
        raise ShadowError("preflight success cannot record candidate cache mutation")
    executed_at = str(report.get("executed_at") or "")
    parsed = datetime.fromisoformat(executed_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ShadowError("preflight executed_at must be timezone-aware UTC")
    if parsed.utcoffset().total_seconds() != 0:
        raise ShadowError("preflight executed_at must be UTC")


def run_preflight(
    *,
    repo_root: Path,
    frozen_config: FrozenShadowConfig,
    official_output_dir: Path,
    preflight_output_dir: Path,
    split: str,
    verify_tag: bool = True,
    require_clean: bool = True,
    verify_runtime: bool = True,
    require_chat_key: bool = True,
) -> dict[str, Any]:
    audit = InputAccessAudit()
    token = _ACCESS_AUDIT.set(audit)
    try:
        resolved_split = canonical_split(split)
        git = git_snapshot(repo_root)
        if require_clean and git["worktree_dirty"]:
            raise ShadowError("structured citation shadow requires a clean worktree")
        if require_clean:
            require_protocol_ancestor(repo_root, str(frozen_config.field("lumenfin_protocol_commit")))
        hashes_before = _readonly_artifact_hashes(repo_root=repo_root, config=frozen_config)
        bind = bind_chain_seal(repo_root=repo_root, config=frozen_config, verify_tag=verify_tag)
        snapshot = None
        if verify_runtime:
            snapshot = verify_runtime_matches_frozen(frozen_config)
        cache = verify_candidate_cache(repo_root=repo_root, config=frozen_config)
        read_sealed_baseline_readonly(repo_root=repo_root, config=frozen_config)
        hashes_after = _readonly_artifact_hashes(repo_root=repo_root, config=frozen_config)
        if hashes_before != hashes_after:
            raise ShadowError("readonly artifact hash changed during preflight")
        expected_manifest = str(frozen_config.field("candidate_cache", "manifest_sha256") or "")
        expected_baseline = str(frozen_config.field("sealed_baseline", "sha256") or "")
        if hashes_after["cache_file_sha256"] != str(cache.get("cache_file_sha256") or ""):
            raise ShadowError("candidate cache file hash mismatch")
        if hashes_after["manifest_sha256"] != expected_manifest:
            raise ShadowError("candidate cache manifest hash mismatch")
        if hashes_after["sealed_baseline_sha256"] != expected_baseline:
            raise ShadowError("sealed baseline hash mismatch")
        prefix_hash = str(cache.get("case_ids_sha256") or "")
        sealed_hash = str(frozen_config.field("case_selection", "query_ids_sha256") or "")
        case_ids_in_sealed_allowlist = bool(prefix_hash) and prefix_hash == sealed_hash
        holdout = audit.prove_holdout_unused(
            split=resolved_split,
            case_ids_in_sealed_allowlist=case_ids_in_sealed_allowlist,
        )
        if official_output_dir.exists():
            raise ShadowError("official shadow output directory already exists")
        if require_chat_key:
            require_chat_credential(repo_root=repo_root)
        credentials = credential_presence(repo_root=repo_root)
        expected_calls = int(frozen_config.field("call_budget", "remote_calls_expected") or 0)
        if preflight_output_dir.exists():
            leftover = [item.name for item in preflight_output_dir.iterdir()]
            if leftover:
                raise ShadowError("preflight output directory already exists")
        report = {
            "schema_version": SCHEMA_VERSION,
            "kind": "preflight",
            "preflight_schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": PREFLIGHT_OK,
            "executed_at": _utc_now(),
            "exit_code": 0,
            "suite": SUITE,
            "split": resolved_split,
            "protocol_ancestor": str(frozen_config.field("lumenfin_protocol_commit")),
            "execution_commit": git["execution_commit"],
            "lumenfin_commit": git["execution_commit"],
            "worktree_status": git["worktree_status"],
            "config_hash": frozen_config.config_hash,
            "seal": bind,
            "candidate_cache": {
                key: value
                for key, value in cache.items()
                if key != "prefix_case_ids"
            },
            "runtime": None if snapshot is None else snapshot.public_dict(),
            "cases_executed": 0,
            "remote_request_count": 0,
            "remote_calls_expected": expected_calls,
            "official_output_dir_exists": False,
            "credentials": credentials,
            "held_out": False,
            "public_holdout_used": holdout["used"],
            "sealed_aggregate_modified": False,
            "candidate_cache_modified": False,
            "product_accuracy_claim": False,
            "benchmark_claim": False,
            "exposed_public_dev_shadow": True,
            "evaluation_mode": EVALUATION_MODE,
            "not_live_production_retrieval": True,
            "candidate_cache_generation": frozen_config.field("candidate_cache_generation"),
            "runtime_components": frozen_config.field("runtime_components"),
            "billing_semantics": BILLING_SEMANTICS,
            "exactly_once": False,
            "integrity": {
                "public_holdout": holdout,
                "sealed_baseline": {
                    "sha256_before": hashes_before["sealed_baseline_sha256"],
                    "sha256_after": hashes_after["sealed_baseline_sha256"],
                    "readonly": True,
                    "modified": False,
                },
                "candidate_cache": {
                    "cache_file_sha256_before": hashes_before["cache_file_sha256"],
                    "cache_file_sha256_after": hashes_after["cache_file_sha256"],
                    "manifest_sha256_before": hashes_before["manifest_sha256"],
                    "manifest_sha256_after": hashes_after["manifest_sha256"],
                    "rebuild": False,
                    "modified": False,
                },
            },
        }
        _assert_preflight_success_contract(report)
        dest = preflight_output_dir / "preflight.json"
        atomic_write_json(dest, report)
        return report
    finally:
        _ACCESS_AUDIT.reset(token)


def _write_run_artifacts(
    *,
    output_dir: Path,
    identity: Mapping[str, Any],
    environment: Mapping[str, Any],
    rows: list[dict[str, Any]],
    summary: Mapping[str, Any],
    sealed: Mapping[str, Any],
) -> None:
    atomic_write_json(output_dir / "manifest.json", identity)
    atomic_write_json(output_dir / "checkpoint.json", identity)
    atomic_write_json(output_dir / "environment.json", environment)
    atomic_write_json(output_dir / "summary.json", summary)
    atomic_write_json(
        output_dir / "citation_accounting.json",
        {
            "schema_version": SCHEMA_VERSION,
            "valid_citations": summary.get("valid_citations"),
            "unknown_citations": summary.get("unknown_citations"),
            "unverified_citations": summary.get("unverified_citations"),
            "cross_scope_citations": summary.get("cross_scope_citations"),
            "stale_citations": summary.get("stale_citations"),
            "citation_validation_failed": summary.get("citation_validation_failed"),
            "citation_source_distribution": summary.get("citation_source_distribution"),
            "product_accuracy_claim": False,
        },
    )
    atomic_write_json(output_dir / "paired_vs_sealed.json", summary.get("paired_vs_sealed") or {})
    failures = [row for row in rows if row.get("failed")]
    atomic_write_text(
        output_dir / "failures.jsonl",
        "".join(json.dumps(sanitize_payload(row), ensure_ascii=False) + "\n" for row in failures),
    )
    atomic_write_text(output_dir / "results.md", render_results_md(summary))


def build_live_generate(snapshot: RuntimeSnapshot) -> GenerateFn:
    from ..llm import DeepSeekChatClient, LLMSettings

    settings = LLMSettings(
        api_key=(os.getenv("DEEPSEEK_API_KEY") or "").strip() or None,
        base_url=snapshot.base_url,
        model=snapshot.model,
        timeout_seconds=snapshot.timeout_seconds,
        max_retries=snapshot.max_retries,
        retry_backoff_seconds=snapshot.retry_backoff_seconds,
    )
    if settings.model != snapshot.model or settings.base_url != snapshot.base_url:
        raise ShadowError("live provider snapshot drifted before initialization")
    client = DeepSeekChatClient(settings)

    def generate(case: Mapping[str, Any]) -> str:
        user_prompt = build_generation_prompt(
            query_text=str(case.get("query_text") or ""),
            hits=list(case.get("hits") or []),
            max_document_chars=4000,
        )
        return client.chat(
            snapshot.prompt,
            user_prompt,
            temperature=0.0,
            max_tokens=200,
        )

    return generate


def run_shadow(
    *,
    repo_root: Path,
    frozen_config: FrozenShadowConfig,
    split: str,
    confirm_exposed_shadow: bool,
    output_dir: Path,
    preflight_output_dir: Path,
    cases: list[dict[str, Any]] | None = None,
    cases_path: Path | None = None,
    allow_remote: bool = False,
    preflight_only: bool = False,
    resume: bool = False,
    generate_fn: GenerateFn | None = None,
    verify_tag: bool = True,
    require_clean: bool = True,
    verify_runtime: bool = True,
    allowlist: list[str] | None = None,
    sealed_override: Mapping[str, Any] | None = None,
    probe: NetworkProbe | None = None,
    allow_injected_generate: bool = False,
    live_generate: bool = False,
    strict_paths: bool = False,
    require_chat_key: bool = True,
) -> dict[str, Any]:
    refuse_env_remote_override()
    if not confirm_exposed_shadow:
        raise ShadowError("structured citation shadow requires --confirm-exposed-shadow")
    canonical_split(split)
    if preflight_only and allow_remote:
        raise ShadowError("refusing --allow-remote with --preflight-only")
    if preflight_only and resume:
        raise ShadowError("refusing --resume with --preflight-only")
    if preflight_only and live_generate:
        raise ShadowError("refusing live generate with --preflight-only")
    if preflight_only:
        if strict_paths:
            assert_exact_output_path(
                output_dir,
                DEFAULT_OFFICIAL_OUTPUT_DIR,
                repo_root=repo_root,
                field="output-dir",
            )
            assert_exact_output_path(
                preflight_output_dir,
                DEFAULT_PREFLIGHT_OUTPUT_DIR,
                repo_root=repo_root,
                field="preflight-dir",
            )
        return run_preflight(
            repo_root=repo_root,
            frozen_config=frozen_config,
            official_output_dir=output_dir,
            preflight_output_dir=preflight_output_dir,
            split=split,
            verify_tag=verify_tag,
            require_clean=require_clean,
            verify_runtime=verify_runtime,
            require_chat_key=require_chat_key,
        )
    if not allow_remote:
        raise ShadowError("formal scoring requires --allow-remote")
    if generate_fn is not None and not allow_injected_generate:
        raise ShadowError("injected generate is not part of the authorized CLI path")
    if generate_fn is None and not live_generate:
        raise ShadowError("live generate is only available through the CLI authorization path")
    if generate_fn is not None and live_generate:
        raise ShadowError("refusing injected generate with live generate")

    if strict_paths:
        assert_exact_output_path(
            output_dir,
            DEFAULT_OFFICIAL_OUTPUT_DIR,
            repo_root=repo_root,
            field="output-dir",
        )
        assert_exact_output_path(
            preflight_output_dir,
            DEFAULT_PREFLIGHT_OUTPUT_DIR,
            repo_root=repo_root,
            field="preflight-dir",
        )

    git = git_snapshot(repo_root)
    if require_clean and git["worktree_dirty"]:
        raise ShadowError("structured citation shadow requires a clean worktree")
    protocol_ancestor = str(frozen_config.field("lumenfin_protocol_commit"))
    if require_clean:
        require_protocol_ancestor(repo_root, protocol_ancestor)
    snapshot = None
    if verify_runtime:
        snapshot = verify_runtime_matches_frozen(frozen_config)
    elif live_generate:
        raise ShadowError("live generate requires a frozen runtime snapshot")
    cache = verify_candidate_cache(repo_root=repo_root, config=frozen_config)
    bind = bind_chain_seal(repo_root=repo_root, config=frozen_config, verify_tag=verify_tag)
    sealed = dict(sealed_override or read_sealed_baseline_readonly(repo_root=repo_root, config=frozen_config))

    expected_hash = str(frozen_config.field("case_selection", "query_ids_sha256"))
    if cases is None:
        if cases_path is None:
            if live_generate:
                raise ShadowError(
                    "live shadow requires bound case payloads; parquet/query text "
                    "is not auto-fetched and cache rebuild is forbidden"
                )
            raise ShadowError("shadow cases fixture is required")
        expected_ids = allowlist or []
        cases = load_case_fixture(
            cases_path,
            allowlist=expected_ids,
            expected_hash=expected_hash,
        )
    else:
        assert_case_ids(
            [str(item["case_id"]) for item in cases],
            expected_ids=allowlist,
            expected_hash=expected_hash,
        )
    if cache.get("prefix_case_ids") and allowlist is None:
        if [str(item["case_id"]) for item in cases] != list(cache["prefix_case_ids"]):
            raise ShadowError("case ids do not match verified candidate cache prefix")

    if live_generate:
        if snapshot is None:
            raise ShadowError("live generate requires a frozen runtime snapshot")
        require_chat_credential(repo_root=repo_root)
        active_generate: GenerateFn = build_live_generate(snapshot)
    elif generate_fn is None:
        raise ShadowError("live generate is only available through the CLI authorization path")
    else:
        active_generate = generate_fn

    require_fresh_or_resume(output_dir, resume=resume)
    output_dir.mkdir(parents=True, exist_ok=True)
    case_manifest_hash = ids_sha256([str(item["case_id"]) for item in cases])
    provider = str(frozen_config.field("chat", "provider"))
    model = str(frozen_config.field("chat", "model"))
    prompt_digest = str(config_field_prompt(frozen_config, snapshot))
    completed_rows: list[dict[str, Any]] = []
    calls_this_invocation = 0
    if resume:
        completed_rows, _manifest, _checkpoint = validate_resume_identity(
            output_dir=output_dir,
            git_commit=str(git["execution_commit"]),
            config_hash=frozen_config.config_hash,
            seal_hash=str(bind["seal_commit"]),
            case_manifest_hash=case_manifest_hash,
            cache_hash=str(cache["cache_file_sha256"]),
            provider=provider,
            model=model,
            protocol_ancestor=protocol_ancestor,
        )
    completed_ids = [str(row["case_id"]) for row in completed_rows]
    remaining = [case for case in cases if str(case["case_id"]) not in set(completed_ids)]
    block_network = not live_generate
    owned_probe = probe is None and block_network
    active_probe = probe or NetworkProbe()
    if owned_probe or (probe is None and block_network):
        active_probe.install()
    try:
        for case in remaining:
            started = time.perf_counter()
            remote_before = active_probe.remote_request_count
            try:
                raw = active_generate(case)
                latency_ms = (time.perf_counter() - started) * 1000.0
                remote_delta = active_probe.remote_request_count - remote_before
                if block_network and remote_delta:
                    raise ShadowError("shadow generate performed a remote request")
                row = score_case(
                    case,
                    raw=raw,
                    latency_ms=latency_ms,
                    generate_attempts=1,
                    remote_calls=1,
                )
            except ShadowError:
                raise
            except Exception as exc:
                row = failed_case(case, exc, remote_calls=1)
            calls_this_invocation += 1
            row["remote_request_count"] = 0 if block_network else int(row.get("remote_calls") or 0)
            row["not_live_production_retrieval"] = True
            append_completed_case(output_dir / "per_case.jsonl", row)
            completed_rows.append(row)
            completed_ids = [str(item["case_id"]) for item in completed_rows]
            calls_total = sum(int(item.get("remote_calls") or 0) for item in completed_rows)
            identity = identity_payload(
                execution_commit=str(git["execution_commit"]),
                protocol_ancestor=protocol_ancestor,
                config_hash=frozen_config.config_hash,
                seal_hash=str(bind["seal_commit"]),
                case_manifest_hash=case_manifest_hash,
                cache_hash=str(cache["cache_file_sha256"]),
                provider=provider,
                model=model,
                prompt_sha256_value=prompt_digest,
                output_dir=output_dir,
                completed_ids=completed_ids,
                cases_total=len(cases),
                calls_expected=int(frozen_config.field("call_budget", "remote_calls_expected") or len(cases)),
                calls_total=calls_total,
                calls_this_invocation=calls_this_invocation,
            )
            atomic_write_json(output_dir / "manifest.json", identity)
            atomic_write_json(output_dir / "checkpoint.json", identity)
        if block_network and active_probe.remote_request_count:
            raise ShadowError("shadow run performed a remote request")
    finally:
        if owned_probe:
            active_probe.remove()

    calls_total = sum(int(item.get("remote_calls") or 0) for item in completed_rows)
    summary = summarize_rows(
        completed_rows,
        cases_total=len(cases),
        remote_request_count=0 if block_network else calls_this_invocation,
        sealed=sealed,
    )
    summary["not_live_production_retrieval"] = True
    identity = identity_payload(
        execution_commit=str(git["execution_commit"]),
        protocol_ancestor=protocol_ancestor,
        config_hash=frozen_config.config_hash,
        seal_hash=str(bind["seal_commit"]),
        case_manifest_hash=case_manifest_hash,
        cache_hash=str(cache["cache_file_sha256"]),
        provider=provider,
        model=model,
        prompt_sha256_value=prompt_digest,
        output_dir=output_dir,
        completed_ids=[str(item["case_id"]) for item in completed_rows],
        cases_total=len(cases),
        calls_expected=int(frozen_config.field("call_budget", "remote_calls_expected") or len(cases)),
        calls_total=calls_total,
        calls_this_invocation=calls_this_invocation,
    )
    environment = {
        "protocol_ancestor": protocol_ancestor,
        "execution_commit": git["execution_commit"],
        "worktree_status": git["worktree_status"],
        "config_hash": frozen_config.config_hash,
        "candidate_cache_hash": cache["cache_file_sha256"],
        "chat_model": model,
        "chat_provider": provider,
        "prompt_sha256": prompt_digest,
        "rerank_provider": frozen_config.field("reranker", "provider"),
        "runtime": None if snapshot is None else snapshot.public_dict(),
        "credentials": credential_presence(repo_root=repo_root),
        "held_out": False,
        "product_accuracy_claim": False,
        "not_live_production_retrieval": True,
    }
    _write_run_artifacts(
        output_dir=output_dir,
        identity=identity,
        environment=environment,
        rows=completed_rows,
        summary=summary,
        sealed=sealed,
    )
    return {
        "summary": summary,
        "identity": identity,
        "cases": completed_rows,
        "remote_request_count": 0 if block_network else calls_this_invocation,
    }


def config_field_prompt(
    frozen_config: FrozenShadowConfig,
    snapshot: RuntimeSnapshot | None,
) -> str:
    if snapshot is not None:
        return prompt_sha256(snapshot.prompt)
    return str(frozen_config.field("prompts", "system_prompt_sha256"))


def parse_cli_guard(argv: list[str] | None = None) -> dict[str, Any]:
    args = list(argv or [])
    forbidden = {
        "--model",
        "--prompt",
        "--rag",
        "--top-k",
        "--top_k",
        "--chunk",
        "--embedding",
        "--reranker",
        "--timeout",
        "--retry",
        "--seed",
    }
    for item in args:
        key = item.split("=", 1)[0]
        if key in forbidden:
            raise ShadowError("shadow CLI refuses runtime parameter overrides")
    return {"argv": args}
