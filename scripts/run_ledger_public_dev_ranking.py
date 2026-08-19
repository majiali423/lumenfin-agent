#!/usr/bin/env python3
"""Run an offline BM25 page-ranking preflight on frozen LEDGER public-dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import bootstrap_dotenv
from lumenfin.eval.financebench.constants import DEFAULT_BM25_RRF_WEIGHT
from lumenfin.eval.financebench.retrieval import build_eval_store
from lumenfin.eval.holdout import (
    ARM_SPECS,
    HoldoutError,
    LedgerPublicDevDataset,
    build_ledger_public_dev_dataset,
    iter_ledger_parquet_rows,
    ledger_public_dev_qrel_audit,
    ledger_snapshot_sha256,
    score_ledger_public_dev,
)
from lumenfin.rag.dashscope_defaults import resolved_dashscope_embedding_model
from lumenfin.rag.hybrid_retriever import reciprocal_rank_fusion
from lumenfin.rag.milvus_store import MilvusRAGStore
from lumenfin.stdio import configure_stdio_utf8

TRACKED_MANIFEST_PATH = (
    ROOT / "data" / "eval_rag" / "holdout" / "ledger_public_manifest.json"
).resolve()
PINNED_MANIFEST_CANONICAL_SHA256 = (
    "e0a3266e3f2492c4f2cdb3e226cdc35d87286fc2f6b0e3bfe432187442932cad"
)
MAX_MONOLITHIC_PREFLIGHT_CASES = 10
MAX_REMOTE_HYBRID_CASES = 100
RUN_CONFIG_SCHEMA = "lumenfin_ledger_ranking_run_config.v2"


def _ids_sha256(values: list[str] | tuple[str, ...]) -> str:
    canonical = "\n".join(sorted(str(value) for value in values))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _evaluator_source_sha256() -> str:
    paths = (
        ROOT / "scripts" / "run_ledger_public_dev_ranking.py",
        ROOT / "scripts" / "run_ledger_public_dev_bm25_sharded.py",
        ROOT / "src" / "lumenfin" / "eval" / "holdout" / "ledger.py",
        ROOT / "src" / "lumenfin" / "eval" / "holdout" / "ledger_corpus.py",
        ROOT / "src" / "lumenfin" / "eval" / "holdout" / "ledger_scoring.py",
        ROOT / "src" / "lumenfin" / "eval" / "holdout" / "page_collapse.py",
        ROOT / "src" / "lumenfin" / "eval" / "holdout" / "ranking.py",
        ROOT / "src" / "lumenfin" / "eval" / "financebench" / "metrics.py",
        ROOT / "src" / "lumenfin" / "eval" / "financebench" / "constants.py",
        ROOT / "src" / "lumenfin" / "eval" / "financebench" / "retrieval.py",
        ROOT / "src" / "lumenfin" / "rag" / "chunking.py",
        ROOT / "src" / "lumenfin" / "rag" / "embeddings.py",
        ROOT / "src" / "lumenfin" / "rag" / "hybrid_retriever.py",
        ROOT / "src" / "lumenfin" / "rag" / "milvus_store.py",
        ROOT / "src" / "lumenfin" / "rag" / "milvus_client.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_run_config(
    manifest: dict,
    *,
    mode: str = "bm25",
    embedding_provider: str = "deterministic",
    embedding_dimension: int = 384,
    embedding_model: str = "",
) -> dict:
    package_versions: dict[str, str] = {}
    for package in ("jieba", "milvus-lite", "pymilvus"):
        try:
            package_versions[package] = version(package)
        except PackageNotFoundError:
            package_versions[package] = "NOT_INSTALLED"
    return {
        "schema_version": RUN_CONFIG_SCHEMA,
        "dataset_snapshot_sha256": manifest["dataset_snapshot_sha256"],
        "manifest_canonical_sha256": PINNED_MANIFEST_CANONICAL_SHA256,
        "evaluator_source_sha256": _evaluator_source_sha256(),
        "arms": {
            name: asdict(spec)
            for name, spec in sorted(ARM_SPECS.items())
        },
        "embedding_provider": embedding_provider,
        "embedding_dimension": embedding_dimension,
        "embedding_model": embedding_model,
        "retrieval_mode": mode,
        "collection_schema": "dense_bm25_v1",
        "qrel_policy": manifest["public_dev_corpus_audit"]["policy"],
        "python_version": sys.version.split()[0],
        "package_versions": package_versions,
    }


def _config_sha256(config: dict) -> str:
    canonical = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score LEDGER public_dev with an isolated BM25 preflight or a one-case "
            "DashScope Hybrid canary. Qwen3 is always disabled."
        )
    )
    parser.add_argument("--parquet-path", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split-salt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--company-key", default="")
    parser.add_argument("--company-shard-index", type=int, default=None)
    parser.add_argument("--company-shard-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--mode", choices=("bm25", "hybrid"), default="bm25")
    parser.add_argument(
        "--embedding-provider",
        choices=("deterministic", "dashscope"),
        default="deterministic",
    )
    parser.add_argument("--embedding-dimension", type=int, default=None)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--allow-remote", action="store_true")
    return parser


def _load_manifest(path: Path) -> dict:
    target = path.expanduser().resolve()
    if target != TRACKED_MANIFEST_PATH:
        raise HoldoutError(
            "LEDGER scoring requires the tracked frozen public benchmark manifest"
        )
    try:
        raw = target.read_bytes()
        payload = json.loads(raw)
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if (
            hashlib.sha256(canonical).hexdigest()
            != PINNED_MANIFEST_CANONICAL_SHA256
        ):
            raise HoldoutError("tracked LEDGER manifest identity has changed")
    except (OSError, json.JSONDecodeError) as exc:
        raise HoldoutError(f"cannot read LEDGER manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise HoldoutError("LEDGER manifest must be an object")
    return payload


def _validate_snapshot_and_salt(
    *,
    manifest: dict,
    snapshot_sha256: str,
    salt: str,
) -> float:
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        raise HoldoutError("LEDGER manifest needs a dataset object")
    if dataset.get("source_artifact_sha256") != snapshot_sha256:
        raise HoldoutError("LEDGER parquet snapshot does not match frozen manifest")
    if manifest.get("public_dev_offline_bm25_preflight_enabled") is not True:
        raise HoldoutError("LEDGER manifest does not enable public_dev BM25 preflight")
    salt_hash = hashlib.sha256(str(salt).encode("utf-8")).hexdigest()
    if manifest.get("split_salt_sha256") != salt_hash:
        raise HoldoutError("LEDGER split salt does not match frozen manifest")
    try:
        fraction = float(manifest["holdout_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HoldoutError("LEDGER manifest holdout_fraction is invalid") from exc
    return fraction


def _dataset_for_queries(
    dataset: LedgerPublicDevDataset,
    queries: tuple[dict, ...],
) -> LedgerPublicDevDataset:
    if not queries:
        raise HoldoutError("LEDGER public_dev preflight selection is empty")
    companies = {str(query["company_key"]) for query in queries}
    documents = tuple(
        document
        for document in dataset.page_documents
        if any(
            company in (document.get("issuer_companies") or [])
            for company in companies
        )
    )
    if not documents:
        raise HoldoutError("LEDGER public_dev preflight corpus is empty")
    return LedgerPublicDevDataset(
        queries=queries,
        page_documents=documents,
        companies=tuple(sorted(companies)),
        reports=len(
            {
                str(document["filename"])
                for document in documents
            }
        ),
    )


def _subset_for_cases(
    dataset: LedgerPublicDevDataset,
    max_cases: int,
) -> LedgerPublicDevDataset:
    if max_cases <= 0:
        raise HoldoutError("--max-cases must be > 0")
    if max_cases > MAX_MONOLITHIC_PREFLIGHT_CASES:
        raise HoldoutError(
            "monolithic LEDGER preflight is limited to 10 cases; "
            "full public_dev requires a company-sharded runner"
        )
    by_company: dict[str, list[dict]] = {}
    for query in dataset.queries:
        by_company.setdefault(str(query["company_key"]), []).append(query)
    selected: list[dict] = []
    offset = 0
    while len(selected) < max_cases:
        added = False
        for company in sorted(by_company):
            company_queries = by_company[company]
            if offset >= len(company_queries):
                continue
            selected.append(company_queries[offset])
            added = True
            if len(selected) >= max_cases:
                break
        if not added:
            break
        offset += 1
    return _dataset_for_queries(dataset, tuple(selected))


def _subset_for_company(
    dataset: LedgerPublicDevDataset,
    *,
    company_key: str,
    max_cases: int,
) -> LedgerPublicDevDataset:
    normalized = company_key.strip().casefold()
    if not normalized or max_cases <= 0:
        raise HoldoutError("company-scoped canary requires a company key and cases > 0")
    queries = tuple(
        query
        for query in dataset.queries
        if str(query["company_key"]).casefold() == normalized
    )[:max_cases]
    if not queries:
        raise HoldoutError("company-scoped canary company is not in public_dev")
    return _dataset_for_queries(dataset, queries)


def _subset_for_company_shard(
    dataset: LedgerPublicDevDataset,
    *,
    shard_index: int,
    shard_count: int,
) -> LedgerPublicDevDataset:
    if shard_count <= 0 or shard_index < 0 or shard_index >= shard_count:
        raise HoldoutError("invalid LEDGER company shard index/count")
    companies = sorted(dataset.companies)
    selected_companies = set(companies[shard_index::shard_count])
    if not selected_companies:
        raise HoldoutError("LEDGER company shard is empty")
    queries = tuple(
        query
        for query in dataset.queries
        if str(query["company_key"]) in selected_companies
    )
    return _dataset_for_queries(dataset, queries)


def _prepare_output_dir(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists():
        marker = target / ".incomplete"
        if (target / "aggregate.json").exists() or not marker.is_file():
            raise HoldoutError(
                f"refusing to reuse existing output directory: {target}"
            )
        for stale_name in (
            "per_case.jsonl",
            "per_case.jsonl.tmp",
            "aggregate.json.tmp",
        ):
            try:
                (target / stale_name).unlink()
            except FileNotFoundError:
                pass
        return target
    if not target.parent.is_dir():
        raise HoldoutError(f"output parent directory not found: {target.parent}")
    target.mkdir()
    (target / ".incomplete").write_text(
        "LEDGER offline preflight has not completed.\n",
        encoding="utf-8",
    )
    return target


def _index_documents(
    store: MilvusRAGStore,
    dataset: LedgerPublicDevDataset,
    *,
    session_id: str,
    batch_size: int,
) -> dict[str, int]:
    if batch_size <= 0:
        raise HoldoutError("--batch-size must be > 0")
    totals = {
        "documents_indexed": 0,
        "chunks_indexed": 0,
        "embed_calls": 0,
        "embed_physical_calls": 0,
        "embed_chars": 0,
        "estimated_dashscope_http_calls": 0,
    }
    documents = list(dataset.page_documents)
    for start in range(0, len(documents), batch_size):
        stats = store.index_documents(
            documents[start : start + batch_size],
            session_id,
        )
        for field in ("documents_indexed", "chunks_indexed", "embed_calls"):
            totals[field] += int(stats.get(field) or 0)
        chunks_indexed = int(stats.get("chunks_indexed") or 0)
        totals["embed_chars"] += int(stats.get("embed_chars") or 0)
        totals["estimated_dashscope_http_calls"] += (
            chunks_indexed + 9
        ) // 10
        totals["embed_physical_calls"] += int(
            getattr(store.embedder, "last_physical_calls", 0)
        )
    if totals["documents_indexed"] != len(documents):
        raise HoldoutError("LEDGER public_dev index document count mismatch")
    if totals["chunks_indexed"] <= 0:
        raise HoldoutError("LEDGER public_dev index contains no chunks")
    return totals


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_write_per_case(path: Path, result: dict) -> None:
    arms = result["per_case"]
    rows = []
    for index in range(result["cases"]):
        rows.append(
            {
                "query_id": arms["A_prod"][index]["case_id"],
                "arms": {
                    arm: arms[arm][index]
                    for arm in ("A_prod", "R_page")
                },
            }
        )
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _retrieve_candidates(
    store: MilvusRAGStore,
    *,
    query: str,
    company: str,
    top_k: int,
    session_id: str,
    mode: str,
) -> tuple[list[dict], dict]:
    bm25_hits = store.bm25_search(
        query,
        session_id=session_id,
        companies=[company],
        top_k=top_k,
    )
    if mode == "bm25":
        return bm25_hits, {
            "mode": "bm25",
            "hits": len(bm25_hits),
            "remote_calls": 0,
        }
    dense_hits = store.vector_search(
        query,
        session_id=session_id,
        companies=[company],
        top_k=top_k,
    )
    query_remote_calls = int(
        getattr(store, "last_query_embed_physical_calls", 0)
    )
    query_cache_hit = bool(
        getattr(store, "last_query_embed_cache_hit", False)
    )
    if query_remote_calls <= 0 and not query_cache_hit:
        raise HoldoutError(
            "hybrid query embedding completed without physical-call accounting"
        )
    if not dense_hits or not bm25_hits:
        raise HoldoutError(
            "hybrid canary requires non-empty Dense and BM25 channels"
        )
    hits = reciprocal_rank_fusion(
        [dense_hits, bm25_hits],
        retrieval_method="hybrid_dense_bm25_rrf",
        weights=[1.0, DEFAULT_BM25_RRF_WEIGHT],
    )[:top_k]
    return hits, {
        "mode": "hybrid_dense_bm25_rrf",
        "bm25_hits": len(bm25_hits),
        "dense_hits": len(dense_hits),
        "query_embedding_cache_hit": query_cache_hit,
        "remote_calls": query_remote_calls,
    }


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    store: MilvusRAGStore | None = None
    output_dir: Path | None = None
    index_dir: Path | None = None
    aggregate: dict = {}
    completed = False
    blocked_error: Exception | None = None
    try:
        mode = str(args.mode)
        embedding_provider = str(args.embedding_provider)
        if mode == "bm25":
            if (
                args.allow_remote
                or embedding_provider != "deterministic"
                or str(args.company_key).strip()
            ):
                raise HoldoutError(
                    "LEDGER BM25 preflight requires deterministic embeddings "
                    "and refuses --allow-remote"
                )
            embedding_dimension = int(args.embedding_dimension or 384)
            if embedding_dimension != 384 or str(args.embedding_model).strip():
                raise HoldoutError(
                    "LEDGER BM25 preflight is locked to deterministic 384-dim embeddings"
                )
            embedding_model = ""
        else:
            if not args.allow_remote or embedding_provider != "dashscope":
                raise HoldoutError(
                    "LEDGER hybrid canary requires dashscope and explicit --allow-remote"
                )
            if (
                args.company_shard_index is not None
                or args.company_shard_count is not None
                or int(args.max_cases) > MAX_REMOTE_HYBRID_CASES
                or not str(args.company_key).strip()
            ):
                raise HoldoutError(
                    "LEDGER remote hybrid canary is limited to 100 cases from "
                    "one explicit company and does not allow company sharding"
                )
            bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
            embedding_dimension = int(args.embedding_dimension or 1024)
            embedding_model = resolved_dashscope_embedding_model(
                str(args.embedding_model or "")
            )
        manifest_path = Path(args.manifest).expanduser().resolve()
        manifest = _load_manifest(manifest_path)
        run_config = _build_run_config(
            manifest,
            mode=mode,
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
            embedding_model=embedding_model,
        )
        run_config_sha256 = _config_sha256(run_config)
        snapshot_hash = ledger_snapshot_sha256(args.parquet_path)
        holdout_fraction = _validate_snapshot_and_salt(
            manifest=manifest,
            snapshot_sha256=snapshot_hash,
            salt=args.split_salt,
        )
        full_dataset = build_ledger_public_dev_dataset(
            iter_ledger_parquet_rows(args.parquet_path),
            salt=args.split_salt,
            holdout_fraction=holdout_fraction,
            manifest=manifest,
        )
        qrel_corpus_audit = ledger_public_dev_qrel_audit(
            full_dataset,
            source_queries=int(manifest["splits"]["public_dev"]["queries"]),
        )
        shard_args = (args.company_shard_index, args.company_shard_count)
        if any(value is not None for value in shard_args):
            if any(value is None for value in shard_args):
                raise HoldoutError(
                    "company shard index and count must be provided together"
                )
            dataset = _subset_for_company_shard(
                full_dataset,
                shard_index=int(args.company_shard_index),
                shard_count=int(args.company_shard_count),
            )
            selection = {
                "strategy": "company_modulo_shard_v1",
                "company_shard_index": int(args.company_shard_index),
                "company_shard_count": int(args.company_shard_count),
                "selected_cases": len(dataset.queries),
                "selected_companies": len(dataset.companies),
            }
        elif str(args.company_key).strip():
            dataset = _subset_for_company(
                full_dataset,
                company_key=str(args.company_key),
                max_cases=int(args.max_cases),
            )
            selection = {
                "strategy": "single_company_prefix_v1",
                "requested_max_cases": int(args.max_cases),
                "selected_cases": len(dataset.queries),
                "selected_companies": len(dataset.companies),
            }
        else:
            dataset = _subset_for_cases(full_dataset, int(args.max_cases))
            selection = {
                "strategy": "company_round_robin_v1",
                "requested_max_cases": int(args.max_cases),
                "selected_cases": len(dataset.queries),
                "selected_companies": len(dataset.companies),
            }
        selection["query_ids_sha256"] = _ids_sha256(
            tuple(str(query["query_id"]) for query in dataset.queries)
        )
        selection["company_keys_sha256"] = _ids_sha256(dataset.companies)
        del full_dataset
        output_dir = _prepare_output_dir(Path(args.output_dir))
        index_dir = output_dir / f"_index_{uuid4().hex[:8]}"
        index_dir.mkdir()
        index_uri = index_dir / "ledger_public_dev.db"
        session_id = (
            f"ledger-public-dev-{manifest['dataset_snapshot_sha256'][:12]}-"
            f"{len(dataset.queries)}"
        )
        store = build_eval_store(
            uri=str(index_uri),
            embedding_provider=embedding_provider,
            embedding_dimension=embedding_dimension,
            collection_name=f"lumenfin_ledger_public_dev_{mode}",
            allow_remote=bool(args.allow_remote),
            mode=mode,
            embedding_model=embedding_model,
        )
        index_stats = _index_documents(
            store,
            dataset,
            session_id=session_id,
            batch_size=int(args.batch_size),
        )

        def retrieve_candidates(query: str, company: str, top_k: int):
            return _retrieve_candidates(
                store,
                query=query,
                company=company,
                top_k=top_k,
                session_id=session_id,
                mode=mode,
            )

        result = score_ledger_public_dev(
            dataset,
            retrieve_candidates=retrieve_candidates,
            retrieval_requires_remote=mode == "hybrid",
            allow_remote=bool(args.allow_remote),
        )
        _atomic_write_per_case(output_dir / "per_case.jsonl", result)
        per_case_sha256 = hashlib.sha256(
            (output_dir / "per_case.jsonl").read_bytes()
        ).hexdigest()
        aggregate = dict(result)
        aggregate.pop("per_case", None)
        aggregate["dataset_snapshot_sha256"] = manifest["dataset_snapshot_sha256"]
        aggregate["run_config"] = run_config
        aggregate["run_config_sha256"] = run_config_sha256
        aggregate["per_case_sha256"] = per_case_sha256
        aggregate["source_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        aggregate["index"] = {
            **index_stats,
            "documents_in_scoped_corpus": len(dataset.page_documents),
            "companies": len(dataset.companies),
            "reports": dataset.reports,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "retrieval_mode": mode,
        }
        aggregate["qrel_corpus_audit"] = qrel_corpus_audit
        aggregate["selection"] = selection
        query_remote_calls = int(
            aggregate["call_accounting"]["retrieval_remote_calls"]
        )
        aggregate["call_accounting"]["document_embedding_remote_calls"] = int(
            index_stats["embed_physical_calls"]
        )
        aggregate["call_accounting"]["query_embedding_remote_calls"] = (
            query_remote_calls
        )
        aggregate["call_accounting"]["remote_calls"] = (
            int(index_stats["embed_physical_calls"]) + query_remote_calls
        )
        aggregate["remote_calls"] = aggregate["call_accounting"]["remote_calls"]
        aggregate["qwen3_calls"] = 0
        _atomic_write_json(output_dir / "aggregate.json", aggregate)
        completed = True
    except Exception as exc:  # noqa: BLE001 - redact all boundary failures
        blocked_error = (
            exc
            if isinstance(exc, (HoldoutError, ValueError))
            else HoldoutError(
                f"LEDGER offline preflight failed: {type(exc).__name__}"
            )
        )
    finally:
        if store is not None:
            try:
                store.close()
            except Exception as exc:  # noqa: BLE001 - cleanup boundary
                completed = False
                blocked_error = HoldoutError(
                    f"LEDGER index close failed: {type(exc).__name__}"
                )
        if not completed and output_dir is not None:
            if index_dir is None or not index_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
            else:
                for stale_name in (
                    "per_case.jsonl",
                    "per_case.jsonl.tmp",
                    "aggregate.json",
                    "aggregate.json.tmp",
                ):
                    try:
                        (output_dir / stale_name).unlink()
                    except FileNotFoundError:
                        pass
    if blocked_error is not None:
        print(f"blocked: {blocked_error}", flush=True)
        return 2
    if output_dir is None:
        print("blocked: LEDGER output directory was not initialized", flush=True)
        return 2
    try:
        (output_dir / ".incomplete").unlink()
    except OSError as exc:
        print(
            f"blocked: LEDGER completion marker cleanup failed: {type(exc).__name__}",
            flush=True,
        )
        return 2
    print(
        "[ledger-public-dev] PREFLIGHT_OK "
        f"cases={aggregate['cases']} "
        f"documents={aggregate['index']['documents_in_scoped_corpus']} "
        f"chunks={aggregate['index']['chunks_indexed']} "
        f"mode={aggregate['index']['retrieval_mode']} "
        f"remote_calls={aggregate['remote_calls']} qwen3_calls=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
