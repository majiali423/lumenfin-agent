"""LEDGER public_holdout E2E v2: selection freeze and unlock helpers.

v1 contract/auth/result stay historical and blocked. This module never opens
query_text, gold values, or qrels during selection freeze.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from .holdout.ledger import (
    PUBLIC_HOLDOUT,
    assign_ledger_company_split,
    ledger_company_key,
)
from .ledger_public_holdout_index import (
    DEFAULT_SNAPSHOT,
    INDEX_DIR,
    PRODUCT_COMMIT,
    PRODUCT_TAG,
    SEAL_PATH,
    SPLIT_SALT,
    SESSION_ID,
    COLLECTION,
    ids_sha256,
    sha256_path,
)

CLAIM = "LEDGER public_holdout held-out end-to-end verified task success"
SUITE = "ledger_public_holdout_held_out_e2e_v2"
SELECTION_STRATEGY = "company_round_robin_v1"
MAX_OFFICIAL_CASES = 100
PARENT_QUERY_IDS_SHA256 = (
    "4457d9268f0eaf14ffff9ab7e847adb9fb787b6283b98874370b3e301047ab88"
)
PARENT_COMPANY_KEYS_SHA256 = (
    "1d5a8e526a6ff682a9cb77288f9c4264a879b5a2719a6e60719a0384d42b3027"
)
IDENTITY_COLUMNS = ("query_id", "exchange", "ticker", "company_name", "year")
FORBIDDEN_SELECTION_COLUMNS = (
    "query_text",
    "value",
    "qrels",
    "kpi",
    "gold",
    "expected",
    "mmd_text",
    "industry",
    "source",
    "tag",
)
SELECTION_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_e2e_selection_v2.json"
CONTRACT_V2_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_e2e_contract_v2.json"
AUTHORIZATION_V2_PATH = (
    Path("data") / "eval_rag" / "ledger_public_holdout_e2e_authorization_v2.json"
)
RESULT_V2_PATH = Path("data") / "eval_rag" / "ledger_public_holdout_e2e_result_v2.json"


class HoldoutE2EV2Error(ValueError):
    """Safe v2 governance error. No holdout text, gold, or secrets."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def canonical_dumps(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def compute_config_hash(payload: Mapping[str, Any]) -> str:
    body = {
        key: value
        for key, value in payload.items()
        if key not in {"config_hash", "authorization_sha256"}
    }
    return sha256_text(canonical_dumps(body))


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    import os

    os.replace(tmp, path)


def iter_holdout_identity_rows(
    snapshot: Path,
    *,
    salt: str = SPLIT_SALT,
) -> Iterator[dict[str, Any]]:
    """Yield holdout identity rows only. Never projects question/gold/qrels/text."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise HoldoutE2EV2Error("pyarrow is required") from exc
    files = sorted(snapshot.rglob("*.parquet")) if snapshot.is_dir() else [snapshot]
    if not files:
        raise HoldoutE2EV2Error("LEDGER parquet snapshot not found")
    for file_path in files:
        handle = parquet.ParquetFile(file_path)
        schema = set(handle.schema_arrow.names)
        missing = [item for item in IDENTITY_COLUMNS if item not in schema]
        if missing:
            raise HoldoutE2EV2Error("snapshot missing identity columns")
        leaked = [item for item in FORBIDDEN_SELECTION_COLUMNS if item in IDENTITY_COLUMNS]
        if leaked:
            raise HoldoutE2EV2Error("identity projection must not include forbidden fields")
        for batch in handle.iter_batches(batch_size=128, columns=list(IDENTITY_COLUMNS)):
            for row in batch.to_pylist():
                if not isinstance(row, dict):
                    raise HoldoutE2EV2Error("projected row is not an object")
                if set(row) - set(IDENTITY_COLUMNS):
                    raise HoldoutE2EV2Error("identity projection leaked fields")
                if set(row).intersection(FORBIDDEN_SELECTION_COLUMNS):
                    raise HoldoutE2EV2Error("identity projection leaked forbidden fields")
                company_key = ledger_company_key(row)
                if assign_ledger_company_split(company_key, salt=salt) != PUBLIC_HOLDOUT:
                    continue
                query_id = str(row.get("query_id") or "").strip()
                if not query_id:
                    raise HoldoutE2EV2Error("holdout identity row missing query_id")
                yield {
                    "query_id": query_id,
                    "company_key": company_key,
                    "exchange": str(row.get("exchange") or ""),
                    "ticker": str(row.get("ticker") or ""),
                    "company_name": str(row.get("company_name") or ""),
                    "year": row.get("year"),
                }


def select_holdout_cases(
    rows: list[dict[str, Any]],
    *,
    max_cases: int = MAX_OFFICIAL_CASES,
) -> list[dict[str, Any]]:
    if max_cases <= 0:
        raise HoldoutE2EV2Error("max_cases must be > 0")
    by_company: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        query_id = str(row["query_id"])
        if query_id in seen:
            continue
        seen.add(query_id)
        by_company[str(row["company_key"])].append(row)
    selected: list[dict[str, Any]] = []
    offset = 0
    while len(selected) < max_cases:
        added = False
        for company in sorted(by_company):
            company_rows = by_company[company]
            if offset >= len(company_rows):
                continue
            selected.append(company_rows[offset])
            added = True
            if len(selected) >= max_cases:
                break
        if not added:
            break
        offset += 1
    if len(selected) != max_cases:
        raise HoldoutE2EV2Error("unable to select the required official case count")
    return selected


def freeze_selection(
    *,
    repo_root: Path,
    snapshot: Path | None = None,
    max_cases: int = MAX_OFFICIAL_CASES,
) -> dict[str, Any]:
    snap = snapshot or (repo_root / DEFAULT_SNAPSHOT)
    rows = list(iter_holdout_identity_rows(snap))
    parent_ids = [str(row["query_id"]) for row in rows]
    parent_companies = sorted({str(row["company_key"]) for row in rows})
    if ids_sha256(parent_ids) != PARENT_QUERY_IDS_SHA256:
        raise HoldoutE2EV2Error("parent holdout query_ids_sha256 mismatch")
    if ids_sha256(parent_companies) != PARENT_COMPANY_KEYS_SHA256:
        raise HoldoutE2EV2Error("parent holdout company_keys_sha256 mismatch")
    selected = select_holdout_cases(rows, max_cases=max_cases)
    selected_ids = [str(item["query_id"]) for item in selected]
    selected_companies = sorted({str(item["company_key"]) for item in selected})
    payload = {
        "schema_version": "ledger_public_holdout_e2e_selection.v2",
        "suite": SUITE,
        "claim": CLAIM,
        "product_tag": PRODUCT_TAG,
        "product_commit": PRODUCT_COMMIT,
        "strategy": SELECTION_STRATEGY,
        "max_official_cases": max_cases,
        "selected_cases": len(selected_ids),
        "selected_companies": len(selected_companies),
        "parent_query_ids_sha256": PARENT_QUERY_IDS_SHA256,
        "parent_company_keys_sha256": PARENT_COMPANY_KEYS_SHA256,
        "selected_query_ids_sha256": ids_sha256(selected_ids),
        "selected_company_keys_sha256": ids_sha256(selected_companies),
        "selected_query_ids": selected_ids,
        "selected_company_keys": selected_companies,
        "holdout_query_text_accessed": False,
        "holdout_gold_accessed": False,
        "holdout_qrels_accessed": False,
        "holdout_consumed": False,
        "identity_columns": list(IDENTITY_COLUMNS),
        "forbidden_columns": list(FORBIDDEN_SELECTION_COLUMNS),
    }
    return payload


def load_index_seal(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / SEAL_PATH
    if not path.exists():
        raise HoldoutE2EV2Error("tracked index seal is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutE2EV2Error("index seal must be an object")
    return payload


def require_sealed_index(*, repo_root: Path) -> dict[str, Any]:
    seal = load_index_seal(repo_root=repo_root)
    if seal.get("status") != "SEALED":
        raise HoldoutE2EV2Error("compatible prebuilt index is not sealed")
    if seal.get("holdout_consumed") is True:
        raise HoldoutE2EV2Error("index seal unexpectedly marks holdout consumed")
    local_db = repo_root / INDEX_DIR / "milvus_lite.db"
    if not local_db.exists():
        raise HoldoutE2EV2Error("sealed Milvus Lite DB is missing locally")
    expected = str(seal.get("milvus_db_sha256") or "")
    if expected and sha256_path(local_db) != expected:
        raise HoldoutE2EV2Error("local Milvus DB sha256 does not match sealed index")
    if seal.get("collection") != COLLECTION:
        raise HoldoutE2EV2Error("sealed collection mismatch")
    if seal.get("session_id") != SESSION_ID:
        raise HoldoutE2EV2Error("sealed session_id mismatch")
    return seal


def build_contract_v2(
    *,
    selection: Mapping[str, Any],
    index_seal: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema_version": "ledger_public_holdout_e2e_contract.v2",
        "kind": "frozen_metric_contract",
        "suite": SUITE,
        "claim": CLAIM,
        "dataset_specific": True,
        "held_out": True,
        "single_use": True,
        "general_product_accuracy_claim": False,
        "product_accuracy_claim": False,
        "financial_accuracy_claim": False,
        "benchmark_claim": False,
        "llm_judge_as_primary": False,
        "product": {
            "release_tag": PRODUCT_TAG,
            "product_commit": PRODUCT_COMMIT,
        },
        "dataset": {
            "split": "public_holdout",
            "manifest_path": "data/eval_rag/holdout/ledger_public_manifest.json",
            "parent_query_ids_sha256": PARENT_QUERY_IDS_SHA256,
            "parent_company_keys_sha256": PARENT_COMPANY_KEYS_SHA256,
            "selection_path": str(SELECTION_PATH).replace("\\", "/"),
            "selection_strategy": SELECTION_STRATEGY,
            "selected_query_ids_sha256": selection["selected_query_ids_sha256"],
            "selected_company_keys_sha256": selection["selected_company_keys_sha256"],
            "source_queries": 2384,
            "source_companies": 26,
            "max_official_cases": MAX_OFFICIAL_CASES,
            "selected_cases": selection["selected_cases"],
        },
        "retrieval": {
            "arm": "A_prod",
            "final_k": 10,
            "source_k": 20,
            "rerank_k": 20,
            "embedding_provider": "dashscope",
            "embedding_dimension": 1024,
            "embedding_model": "text-embedding-v4",
            "reranker": "qwen3-rerank",
            "candidate_cache_forbidden": True,
            "public_dev_index_forbidden": True,
            "document_reembedding_calls": 0,
            "requires_compatible_prebuilt_index": True,
            "index_collection": COLLECTION,
            "index_session_id": SESSION_ID,
            "index_uri": str(index_seal.get("uri") or "").replace("\\", "/"),
            "index_milvus_db_sha256": index_seal.get("milvus_db_sha256"),
        },
        "generation": {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "citation_alias_protocol": "citation_alias_protocol.v1",
            "structured_answer_schema": "1.0",
            "scorer_version": "lumenfin_ledger_e2e_scoring.v1",
            "relative_tolerance": 0.01,
        },
        "primary_metric": {
            "name": "strict_verified_e2e_success_rate",
            "verified_e2e_success": [
                "answer_correct",
                "structured_contract_valid",
                "citation_supported",
                "provider_success",
            ],
            "denominator": "all_selected_holdout_cases",
            "empty_qrels": "NOT_EVALUABLE",
            "confidence_interval": "wilson_95",
        },
        "call_budget": {
            "holdout_cases": MAX_OFFICIAL_CASES,
            "deepseek_logical_calls": MAX_OFFICIAL_CASES,
            "query_embedding_calls": MAX_OFFICIAL_CASES,
            "qwen3_rerank_calls": MAX_OFFICIAL_CASES,
            "document_reembedding_calls": 0,
        },
        "index_gate": {
            "compatible_prebuilt_index_present": True,
            "reason": "sealed_public_holdout_rc5_index_v1",
        },
    }
    payload["config_hash"] = compute_config_hash(payload)
    return payload


PREFLIGHT_DIR = Path("outputs") / "ledger_public_holdout_e2e_preflight_v2"
OFFICIAL_DIR = Path("outputs") / "ledger_public_holdout_e2e_v2"
CASE_COLUMNS = (
    "query_id",
    "query_text",
    "value",
    "qrels",
    "exchange",
    "ticker",
    "company_name",
    "year",
)
FORBIDDEN_CLI = {
    "--dataset",
    "--dataset-path",
    "--cases",
    "--cases-path",
    "--parquet",
    "--parquet-path",
    "--split",
    "--limit",
    "--model",
    "--prompt",
    "--final-k",
    "--final_k",
    "--output-dir",
    "--preflight-dir",
    "--embedding",
    "--reranker",
    "--ignore-qrels",
    "--skip-citation-validation",
    "--candidate-dir",
    "--config",
    "--config-file",
    "--config-hash",
    "--index",
    "--index-uri",
    "--collection",
}
AUTHORIZING_ENV = (
    "LEDGER_PUBLIC_HOLDOUT_E2E_FORCE",
    "LEDGER_HOLDOUT_E2E_FORCE",
    "MAS_ALLOW_HOLDOUT_RERUN",
    "MAS_HOLDOUT_FORCE",
    "HOLDOUT_E2E_ALLOW_RERUN",
)
OFFICIAL_RAW_RELATIVE_PATHS = (
    "outputs/ledger_public_holdout_e2e_v2/aggregate.json",
    "outputs/ledger_public_holdout_e2e_v2/per_case.jsonl",
)
RESULT_KIND = "ledger_public_holdout_e2e_result_ledger"
RESULT_SCHEMA_VERSION = "ledger_public_holdout_e2e_result.v2"
EXPECTED_CASES_TOTAL = 100
EXPECTED_SUCCESSES = 35
EXPECTED_RATE = 0.35
EXPECTED_WILSON_DISPLAY = (0.264, 0.447)
EXPECTED_ANSWER_CORRECT = 37
EXPECTED_STRUCTURED_CONTRACT_VALID = 47
EXPECTED_CITATION_SUPPORTED = 40
EXPECTED_PROVIDER_SUCCESS = 100
EXPECTED_ABSTAIN = 54
EXPECTED_CONFIG_HASH = (
    "233605b6f1d047ea8cd0d2a67a0e912da5df193cde8b909c338c4a424e4de01d"
)
EXPECTED_INDEX_HASH = (
    "10cdfdddeb0fdb069cc9aaaef0fde71fa6b7c09501c80f743ce40eff8693fc83"
)
EXPECTED_SELECTED_QUERY_IDS_SHA256 = (
    "c2f54a7ffe9fcc8f50946dc3af9bc5efb8ec8d01e049cbf20d481e3026b72d31"
)
EXPECTED_SELECTION_FILE_SHA256 = (
    "cf2d3355843236943365033b5c7c4658a51172cfd6c6aa588f19b1afeb09b55b"
)
EXPECTED_OFFICIAL_RAW_ARTIFACTS = (
    {
        "relative_path": "outputs/ledger_public_holdout_e2e_v2/aggregate.json",
        "size_bytes": 1548,
        "sha256": "eb25ed7f5618c0f689707165db3a9b682028e6f35985c377de84111150f7fb17",
    },
    {
        "relative_path": "outputs/ledger_public_holdout_e2e_v2/per_case.jsonl",
        "size_bytes": 97358,
        "sha256": "c7c4e138751e10cf763e63b5358746fb0fdb2a13fcec08aa734e6781b931ad57",
    },
)
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
LEAK_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._\-]{8,}"),
    re.compile(r"DASHSCOPE_API_KEY\s*="),
    re.compile(r"DEEPSEEK_API_KEY\s*="),
    re.compile(r"MAS_API_KEY\s*="),
    re.compile(r"(?i)authorization\s*:\s*\S+"),
    re.compile(r"[A-Za-z]:\\Users\\"),
    re.compile(r"/Users/[A-Za-z0-9._-]+"),
)


def load_selection(*, repo_root: Path, allow_consumed: bool = False) -> dict[str, Any]:
    path = repo_root / SELECTION_PATH
    if not path.exists():
        raise HoldoutE2EV2Error("selection freeze is missing; run freeze script --write")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutE2EV2Error("selection must be an object")
    if payload.get("schema_version") != "ledger_public_holdout_e2e_selection.v2":
        raise HoldoutE2EV2Error("selection schema mismatch")
    if int(payload.get("selected_cases") or 0) != MAX_OFFICIAL_CASES:
        raise HoldoutE2EV2Error("selection case count mismatch")
    if payload.get("holdout_consumed") is True and not allow_consumed:
        raise HoldoutE2EV2Error("selection unexpectedly marks holdout consumed")
    ids = [str(item) for item in (payload.get("selected_query_ids") or [])]
    if ids_sha256(ids) != payload.get("selected_query_ids_sha256"):
        raise HoldoutE2EV2Error("selection query_ids sha256 mismatch")
    return payload


def load_contract_v2(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / CONTRACT_V2_PATH
    if not path.exists():
        raise HoldoutE2EV2Error("contract_v2 is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutE2EV2Error("contract_v2 must be an object")
    if payload.get("schema_version") != "ledger_public_holdout_e2e_contract.v2":
        raise HoldoutE2EV2Error("contract_v2 schema mismatch")
    if payload.get("claim") != CLAIM:
        raise HoldoutE2EV2Error("contract_v2 claim mismatch")
    if payload.get("llm_judge_as_primary") is not False:
        raise HoldoutE2EV2Error("LLM judge cannot be the primary metric")
    if payload.get("general_product_accuracy_claim") is not False:
        raise HoldoutE2EV2Error("general product accuracy claim must stay false")
    product = payload.get("product") or {}
    if product.get("product_commit") != PRODUCT_COMMIT or product.get("release_tag") != PRODUCT_TAG:
        raise HoldoutE2EV2Error("contract_v2 is not locked to v0.1.0-rc.5")
    gate = payload.get("index_gate") or {}
    if gate.get("compatible_prebuilt_index_present") is not True:
        raise HoldoutE2EV2Error("contract_v2 index gate must be open")
    computed = compute_config_hash(payload)
    stored = str(payload.get("config_hash") or "")
    if stored and stored != computed:
        raise HoldoutE2EV2Error("contract_v2 config_hash mismatch")
    payload["config_hash"] = computed
    return payload


def build_authorization_v2(*, contract: Mapping[str, Any]) -> dict[str, Any]:
    config_hash = str(contract["config_hash"])
    return {
        "schema_version": "ledger_public_holdout_e2e_authorization.v2",
        "kind": "ledger_public_holdout_e2e_authorization",
        "default_deny": True,
        "records": {
            config_hash: {
                "config_hash": config_hash,
                "identity_status": "ONE_SHOT_AUTHORIZED",
                "execution_authorized": True,
                "preflight_authorized": True,
                "remote_run_authorized": True,
                "reason": "sealed_public_holdout_rc5_index_v1",
                "official_preflight_executions": 0,
                "official_remote_executions": 0,
                "dataset_consumed": False,
                "holdout_consumed": False,
            }
        },
    }


def load_authorization_v2(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / AUTHORIZATION_V2_PATH
    if not path.exists():
        raise HoldoutE2EV2Error("authorization_v2 is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutE2EV2Error("authorization_v2 must be an object")
    if payload.get("kind") != "ledger_public_holdout_e2e_authorization":
        raise HoldoutE2EV2Error("authorization kind mismatch")
    if payload.get("default_deny") is not True:
        raise HoldoutE2EV2Error("authorization must be default-deny")
    if payload.get("schema_version") != "ledger_public_holdout_e2e_authorization.v2":
        raise HoldoutE2EV2Error("authorization_v2 schema mismatch")
    return payload


def execution_authorized_v2(
    contract: Mapping[str, Any],
    *,
    repo_root: Path,
    want: str,
) -> bool:
    auth = load_authorization_v2(repo_root=repo_root)
    record = (auth.get("records") or {}).get(str(contract["config_hash"]))
    if not isinstance(record, Mapping):
        return False
    if record.get("dataset_consumed") is True or record.get("holdout_consumed") is True:
        return False
    flags = {
        "preflight": "preflight_authorized",
        "remote": "remote_run_authorized",
    }
    flag = flags.get(want)
    if flag is None:
        return False
    return (
        record.get("execution_authorized") is True
        and record.get(flag) is True
        and record.get("identity_status") == "ONE_SHOT_AUTHORIZED"
    )


def authorization_record_v2(
    *,
    repo_root: Path,
    contract: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    loaded = contract or load_contract_v2(repo_root=repo_root)
    auth = load_authorization_v2(repo_root=repo_root)
    record = (auth.get("records") or {}).get(str(loaded["config_hash"]))
    return dict(record) if isinstance(record, Mapping) else None


def holdout_consumed_v2(
    *,
    repo_root: Path,
    contract: Mapping[str, Any] | None = None,
) -> bool:
    record = authorization_record_v2(repo_root=repo_root, contract=contract)
    if record and (
        record.get("holdout_consumed") is True or record.get("dataset_consumed") is True
    ):
        return True
    selection_path = repo_root / SELECTION_PATH
    if selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if isinstance(selection, dict) and selection.get("holdout_consumed") is True:
            return True
    result_path = repo_root / RESULT_V2_PATH
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if isinstance(result, dict) and (
            result.get("holdout_consumed") is True or result.get("seal_status") == "SEALED"
        ):
            return True
    return False


def refuse_runtime_overrides(argv: list[str] | None = None) -> None:
    args = list(argv or [])
    for name in AUTHORIZING_ENV:
        if (os.getenv(name) or "").strip():
            raise HoldoutE2EV2Error("environment cannot authorize holdout execution")
    for flag in FORBIDDEN_CLI:
        if flag in args:
            raise HoldoutE2EV2Error("CLI refuses runtime overrides")


def write_contract_and_auth(*, repo_root: Path) -> dict[str, Any]:
    if (repo_root / AUTHORIZATION_V2_PATH).exists() and holdout_consumed_v2(repo_root=repo_root):
        raise HoldoutE2EV2Error("holdout already consumed; refusing rewrite")
    if (repo_root / SELECTION_PATH).exists():
        existing_selection = json.loads((repo_root / SELECTION_PATH).read_text(encoding="utf-8"))
        if isinstance(existing_selection, dict) and existing_selection.get("holdout_consumed") is True:
            raise HoldoutE2EV2Error("holdout already consumed; refusing rewrite")
    selection = (
        load_selection(repo_root=repo_root) if (repo_root / SELECTION_PATH).exists() else None
    )
    if selection is None:
        selection = freeze_selection(repo_root=repo_root)
        atomic_write_json(repo_root / SELECTION_PATH, selection)
    seal = require_sealed_index(repo_root=repo_root)
    contract = build_contract_v2(selection=selection, index_seal=seal)
    auth = build_authorization_v2(contract=contract)
    if (repo_root / CONTRACT_V2_PATH).exists():
        existing = json.loads((repo_root / CONTRACT_V2_PATH).read_text(encoding="utf-8"))
        if existing.get("config_hash") != contract["config_hash"]:
            raise HoldoutE2EV2Error("refusing to overwrite a different contract_v2")
    if (repo_root / AUTHORIZATION_V2_PATH).exists():
        existing_auth = load_authorization_v2(repo_root=repo_root)
        record = (existing_auth.get("records") or {}).get(str(contract["config_hash"]))
        if isinstance(record, Mapping) and record.get("holdout_consumed") is True:
            raise HoldoutE2EV2Error("holdout already consumed; refusing rewrite")
    atomic_write_json(repo_root / CONTRACT_V2_PATH, contract)
    atomic_write_json(repo_root / AUTHORIZATION_V2_PATH, auth)
    return {"selection": selection, "contract": contract, "authorization": auth, "index_seal": seal}


def refuse_unauthorized_v2(*, repo_root: Path, argv: list[str] | None = None) -> None:
    """Fail closed before credentials, holdout text, providers, dirs, or remotes."""
    args = list(argv or [])
    refuse_runtime_overrides(args)
    if "--confirm-public-holdout-consumption" not in args and any(
        token in " ".join(args).casefold() for token in ("public_holdout", "public-holdout")
    ):
        raise HoldoutE2EV2Error("CLI refuses holdout path without consumption confirm")
    contract = load_contract_v2(repo_root=repo_root)
    if holdout_consumed_v2(repo_root=repo_root, contract=contract):
        raise HoldoutE2EV2Error("remote run is not authorized or already consumed")
    if "--preflight-only" in args:
        if not execution_authorized_v2(contract, repo_root=repo_root, want="preflight"):
            raise HoldoutE2EV2Error("preflight is not authorized")
        require_sealed_index(repo_root=repo_root)
        return
    if "--confirm-public-holdout-consumption" not in args:
        raise HoldoutE2EV2Error("official consume requires --confirm-public-holdout-consumption")
    if "--allow-remote" not in args:
        raise HoldoutE2EV2Error("official consume requires --allow-remote")
    if not execution_authorized_v2(contract, repo_root=repo_root, want="remote"):
        raise HoldoutE2EV2Error("remote run is not authorized or already consumed")
    require_sealed_index(repo_root=repo_root)


def credential_status() -> dict[str, Any]:
    import os

    dash = bool((os.getenv("DASHSCOPE_API_KEY") or "").strip())
    deep = bool((os.getenv("DEEPSEEK_API_KEY") or "").strip())
    return {
        "dashscope_api_key_present": dash,
        "deepseek_api_key_present": deep,
        "ready": dash and deep,
    }


def build_preflight_payload(
    *,
    contract: Mapping[str, Any],
    selection: Mapping[str, Any],
    index_seal: Mapping[str, Any],
) -> dict[str, Any]:
    cred = credential_status()
    return {
        "status": "PREFLIGHT_OK" if cred["ready"] else "PREFLIGHT_BLOCKED",
        "schema_version": "ledger_public_holdout_e2e_preflight.v2",
        "exit_code": 0 if cred["ready"] else 2,
        "cases_executed": 0,
        "cases_total": 0,
        "remote_request_count": 0,
        "holdout_content_logged": False,
        "holdout_consumed": False,
        "holdout_query_text_accessed": False,
        "holdout_gold_accessed": False,
        "holdout_qrels_accessed": False,
        "product_commit": PRODUCT_COMMIT,
        "config_hash": contract["config_hash"],
        "selected_query_ids_sha256": selection["selected_query_ids_sha256"],
        "index_collection": index_seal.get("collection"),
        "index_session_id": index_seal.get("session_id"),
        "index_milvus_db_sha256": index_seal.get("milvus_db_sha256"),
        "document_reembedding_calls": 0,
        "public_dev_candidate_cache_used": False,
        "credentials": {
            "dashscope_api_key_present": cred["dashscope_api_key_present"],
            "deepseek_api_key_present": cred["deepseek_api_key_present"],
        },
    }


def mark_auth_execution(
    *,
    repo_root: Path,
    config_hash: str,
    kind: str,
) -> dict[str, Any]:
    auth = load_authorization_v2(repo_root=repo_root)
    record = dict((auth.get("records") or {}).get(config_hash) or {})
    if not record:
        raise HoldoutE2EV2Error("authorization record missing for config_hash")
    if record.get("holdout_consumed") is True:
        raise HoldoutE2EV2Error("holdout already consumed")
    if kind == "preflight":
        record["official_preflight_executions"] = int(record.get("official_preflight_executions") or 0) + 1
    elif kind == "remote":
        record["official_remote_executions"] = int(record.get("official_remote_executions") or 0) + 1
        record["holdout_consumed"] = True
        record["dataset_consumed"] = True
        record["identity_status"] = "CONSUMED"
        record["execution_authorized"] = False
        record["preflight_authorized"] = False
        record["remote_run_authorized"] = False
    else:
        raise HoldoutE2EV2Error("unknown authorization mark kind")
    auth = dict(auth)
    records = dict(auth.get("records") or {})
    records[config_hash] = record
    auth["records"] = records
    atomic_write_json(repo_root / AUTHORIZATION_V2_PATH, auth)
    return auth


def load_selected_cases(
    *,
    repo_root: Path,
    selection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Official consume only. Opens query_text, gold, and qrels for selected IDs."""
    from .holdout.governance import HoldoutError
    from .holdout.ledger import ledger_company_key
    from .holdout.ledger import _normalize_qrels  # noqa: PLC2701 - eval-only loader

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise HoldoutE2EV2Error("pyarrow is required") from exc

    wanted = [str(item) for item in selection["selected_query_ids"]]
    wanted_set = set(wanted)
    snapshot = repo_root / DEFAULT_SNAPSHOT
    files = sorted(snapshot.rglob("*.parquet")) if snapshot.is_dir() else [snapshot]
    if not files:
        raise HoldoutE2EV2Error("LEDGER parquet snapshot not found")
    found: dict[str, dict[str, Any]] = {}
    for file_path in files:
        handle = parquet.ParquetFile(file_path)
        names = set(handle.schema_arrow.names)
        missing = [col for col in CASE_COLUMNS if col not in names]
        if missing:
            raise HoldoutE2EV2Error("snapshot missing official case columns")
        for batch in handle.iter_batches(batch_size=64, columns=list(CASE_COLUMNS)):
            for row in batch.to_pylist():
                query_id = str(row.get("query_id") or "").strip()
                if query_id not in wanted_set or query_id in found:
                    continue
                try:
                    qrels = _normalize_qrels(row.get("qrels"))
                    gold = float(row.get("value"))
                except (HoldoutError, TypeError, ValueError) as exc:
                    raise HoldoutE2EV2Error(f"selected case {query_id} is invalid") from exc
                company_key = ledger_company_key(row)
                found[query_id] = {
                    "query_id": query_id,
                    "case_id": query_id,
                    "query_text": str(row.get("query_text") or "").strip(),
                    "company_key": company_key,
                    "exchange": str(row.get("exchange") or ""),
                    "ticker": str(row.get("ticker") or ""),
                    "company_name": str(row.get("company_name") or ""),
                    "year": row.get("year"),
                    "gold_value": gold,
                    "qrels": {str(item["doc_id"]): int(item["relevance"]) for item in qrels},
                    "tenant_id": SESSION_ID,
                    "session_id": SESSION_ID,
                }
                if not found[query_id]["query_text"]:
                    raise HoldoutE2EV2Error(f"selected case {query_id} missing query_text")
                if len(found) == len(wanted_set):
                    break
            if len(found) == len(wanted_set):
                break
        if len(found) == len(wanted_set):
            break
    missing_ids = [query_id for query_id in wanted if query_id not in found]
    if missing_ids:
        raise HoldoutE2EV2Error("selected holdout cases are incomplete in the snapshot")
    return [found[query_id] for query_id in wanted]


def sha256_file_bytes(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson_display(low: float, high: float) -> list[float]:
    return [round(float(low), 3), round(float(high), 3)]


def load_result_v2(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / RESULT_V2_PATH
    if not path.exists():
        raise HoldoutE2EV2Error("result_v2 is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HoldoutE2EV2Error("result_v2 must be an object")
    return payload


def scan_text_for_leaks(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in LEAK_PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    if re.search(r'"query_text"\s*:\s*"', text):
        hits.append("query_text_body")
    if re.search(r'"gold_value"\s*:', text) or re.search(r'"gold"\s*:\s*-?[0-9]', text):
        hits.append("gold_body")
    if re.search(r'"qrels"\s*:\s*\{', text):
        hits.append("qrels_body")
    return hits


def inspect_official_raw_artifacts(*, repo_root: Path) -> dict[str, Any]:
    official_dir = repo_root / OFFICIAL_DIR
    if not official_dir.exists():
        return {
            "raw_artifacts_status": "NOT_PRESENT",
            "raw_bytes_reverified": False,
            "artifacts": [],
            "extra_files": [],
        }
    present: list[dict[str, Any]] = []
    extra: list[str] = []
    expected = {item.replace("\\", "/") for item in OFFICIAL_RAW_RELATIVE_PATHS}
    for path in sorted(p for p in official_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(repo_root).as_posix()
        if rel not in expected:
            extra.append(rel)
            continue
        present.append(
            {
                "relative_path": rel,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file_bytes(path),
            }
        )
    return {
        "raw_artifacts_status": "PRESENT",
        "raw_bytes_reverified": False,
        "artifacts": present,
        "extra_files": extra,
    }


def tally_official_per_case(*, repo_root: Path) -> dict[str, Any]:
    path = repo_root / OFFICIAL_DIR / "per_case.jsonl"
    if not path.exists():
        return {"rows": 0, "query_ids": [], "remaining": None}
    query_ids: list[str] = []
    counts = {
        "verified_e2e_success": 0,
        "answer_correct": 0,
        "structured_contract_valid": 0,
        "citation_supported": 0,
        "provider_success": 0,
        "abstain": 0,
        "provider_errors": 0,
    }
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise HoldoutE2EV2Error("per-case jsonl row must be an object")
        query_id = str(row.get("query_id") or "").strip()
        if not query_id:
            raise HoldoutE2EV2Error("per-case row missing query_id")
        query_ids.append(query_id)
        for key in (
            "verified_e2e_success",
            "answer_correct",
            "structured_contract_valid",
            "citation_supported",
            "provider_success",
            "abstain",
        ):
            if row.get(key) is True:
                counts[key] += 1
        if row.get("provider_success") is False:
            counts["provider_errors"] += 1
    unique = list(dict.fromkeys(query_ids))
    return {
        "rows": len(query_ids),
        "unique_query_ids": len(unique),
        "query_ids": unique,
        "remaining": 0 if len(unique) == EXPECTED_CASES_TOTAL else EXPECTED_CASES_TOTAL - len(unique),
        **counts,
    }


def reverify_raw_artifacts_against_seal(
    *,
    repo_root: Path,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    inspection = inspect_official_raw_artifacts(repo_root=repo_root)
    if inspection["raw_artifacts_status"] == "NOT_PRESENT":
        return {
            "raw_artifacts_status": "NOT_PRESENT",
            "raw_bytes_reverified": False,
        }
    sealed = result or load_result_v2(repo_root=repo_root)
    expected = list(sealed.get("official_raw_artifacts") or EXPECTED_OFFICIAL_RAW_ARTIFACTS)
    by_path = {str(item["relative_path"]): item for item in expected}
    mismatches: list[str] = []
    if inspection["extra_files"]:
        mismatches.append("extra_official_files")
    observed = {item["relative_path"]: item for item in inspection["artifacts"]}
    for rel, item in by_path.items():
        got = observed.get(rel)
        if got is None:
            mismatches.append(f"missing:{rel}")
            continue
        if int(got["size_bytes"]) != int(item["size_bytes"]):
            mismatches.append(f"size:{rel}")
        if str(got["sha256"]) != str(item["sha256"]):
            mismatches.append(f"sha256:{rel}")
    ok = not mismatches
    return {
        "raw_artifacts_status": "PRESENT",
        "raw_bytes_reverified": ok,
        "mismatches": mismatches,
        "artifacts": inspection["artifacts"],
        "extra_files": inspection["extra_files"],
    }
