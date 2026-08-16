from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from ...logging_utils import redact_secrets
from ...provider_resilience import redact_provider_message
from .constants import SCHEMA_VERSION
from .metrics import bootstrap_mean_ci, mean
from .paired import compare_paired_systems


_SECRET_KEY_RE = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|token|password|secret|access[_-]?key|request[_-]?id|endpoint|base_url)"
)
_SK_RE = re.compile(r"\bsk-[A-Za-z0-9_-]+")
_HTTP_RE = re.compile(r"https?://[^\s\"']+", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        "lumenfin_commit": commit or "unknown",
        "worktree_dirty": bool(porcelain),
        "worktree_status": "dirty" if porcelain else "clean",
    }


def redact_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_RE.search(str(key)):
                cleaned[str(key)] = "[REDACTED]"
                continue
            cleaned[str(key)] = redact_mapping(item)
        return cleaned
    if isinstance(value, list):
        return [redact_mapping(item) for item in value]
    if isinstance(value, str):
        text = str(redact_secrets(value))
        text = redact_provider_message(text, limit=max(300, len(text) + 1))
        text = _SK_RE.sub("[REDACTED]", text)
        text = _HTTP_RE.sub("[REDACTED_URL]", text)
        return text
    return value


def assert_no_secrets(payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    if _SK_RE.search(serialized):
        raise ValueError("evaluation output contains an API key-like token")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cleaned = redact_mapping(payload)
    assert_no_secrets(cleaned)
    path.write_text(json.dumps(cleaned, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as handle:
        for row in rows:
            cleaned = redact_mapping(dict(row))
            assert_no_secrets(cleaned)
            handle.write(json.dumps(cleaned, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def completed_case_ids(path: Path, *, mode: str) -> set[str]:
    done: set[str] = set()
    for row in read_jsonl(path):
        if str(row.get("mode") or "") == mode and row.get("case_id"):
            done.add(str(row["case_id"]))
    return done


def environment_payload(
    *,
    repo_root: Path,
    dataset_hash: str,
    split_manifest_hash: str,
    embedding_provider: str,
    embedding_model: str,
    rerank_provider: str,
    rerank_model: str,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
    bm25_rrf_weight: float,
    top_k: int,
    mode: str,
    split: str,
    remote_calls_enabled: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot = git_snapshot(repo_root)
    payload = {
        "schema_version": SCHEMA_VERSION,
        **snapshot,
        "dataset_hash": dataset_hash,
        "split_manifest_hash": split_manifest_hash,
        "python_version": platform.python_version(),
        "embedding_provider": embedding_provider,
        "embedding_model": embedding_model,
        "rerank_provider": rerank_provider,
        "rerank_model": rerank_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "collection_name": collection_name,
        "bm25_rrf_weight": bm25_rrf_weight,
        "top_k": top_k,
        "mode": mode,
        "split": split,
        "executed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "remote_calls_enabled": bool(remote_calls_enabled),
        "status": "NOT_RUN" if extra and extra.get("status") == "NOT_RUN" else "recorded",
    }
    if extra:
        payload.update(extra)
    return redact_mapping(payload)


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if q <= 0:
        return ordered[0]
    if q >= 1:
        return ordered[-1]
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[index]


def aggregate_case_metrics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {"cases": 0, "status": "EMPTY"}
    page = {
        "hit_at_1": mean(float(item["page"]["hit_at"]["1"]) for item in cases),
        "hit_at_3": mean(float(item["page"]["hit_at"]["3"]) for item in cases),
        "hit_at_5": mean(float(item["page"]["hit_at"]["5"]) for item in cases),
        "hit_at_10": mean(float(item["page"]["hit_at"]["10"]) for item in cases),
        "recall_at_1": mean(float(item["page"]["recall_at"]["1"]) for item in cases),
        "recall_at_3": mean(float(item["page"]["recall_at"]["3"]) for item in cases),
        "recall_at_5": mean(float(item["page"]["recall_at"]["5"]) for item in cases),
        "recall_at_10": mean(float(item["page"]["recall_at"]["10"]) for item in cases),
        "mrr": mean(float(item["page"]["mrr"]) for item in cases),
        "ndcg_at_5": mean(float(item["page"]["ndcg_at"]["5"]) for item in cases),
        "ndcg_at_10": mean(float(item["page"]["ndcg_at"]["10"]) for item in cases),
    }
    chunk = {
        "hit_at_5": mean(float(item["chunk"]["hit_at"]["5"]) for item in cases),
        "hit_at_10": mean(float(item["chunk"]["hit_at"]["10"]) for item in cases),
        "hit_at_20": mean(float(item["chunk"]["hit_at"]["20"]) for item in cases),
        "recall_at_5": mean(float(item["chunk"]["recall_at"]["5"]) for item in cases),
        "recall_at_10": mean(float(item["chunk"]["recall_at"]["10"]) for item in cases),
        "recall_at_20": mean(float(item["chunk"]["recall_at"]["20"]) for item in cases),
        "mrr": mean(float(item["chunk"]["mrr"]) for item in cases),
        "ndcg_at_10": mean(float(item["chunk"]["ndcg_at"]["10"]) for item in cases),
        "deprecated": True,
        "semantics": "union of page-derived and span-overlap chunk ids",
    }

    def _chunk_family(key: str) -> dict[str, float]:
        present = [item for item in cases if isinstance(item.get(key), dict)]
        if not present:
            return {}
        return {
            "hit_at_5": round(mean(float(item[key]["hit_at"]["5"]) for item in present), 4),
            "hit_at_10": round(mean(float(item[key]["hit_at"]["10"]) for item in present), 4),
            "hit_at_20": round(mean(float(item[key]["hit_at"]["20"]) for item in present), 4),
            "recall_at_5": round(mean(float(item[key]["recall_at"]["5"]) for item in present), 4),
            "recall_at_10": round(mean(float(item[key]["recall_at"]["10"]) for item in present), 4),
            "recall_at_20": round(mean(float(item[key]["recall_at"]["20"]) for item in present), 4),
            "mrr": round(mean(float(item[key]["mrr"]) for item in present), 4),
            "ndcg_at_10": round(mean(float(item[key]["ndcg_at"]["10"]) for item in present), 4),
        }

    page_chunk = _chunk_family("page_chunk")
    span_chunk = _chunk_family("span_chunk")
    mapped = sum(int(item.get("span_mapped_count") or 0) for item in cases)
    unmapped = sum(int(item.get("span_unmapped_count") or 0) for item in cases)
    single_gold = all(bool(item.get("single_gold_page")) for item in cases)
    return {
        "cases": len(cases),
        "succeeded": sum(1 for item in cases if item.get("status") == "ok"),
        "failed": sum(1 for item in cases if item.get("status") != "ok"),
        "single_gold_page_all": single_gold,
        "recall_equals_hit_note": (
            "此时 Recall@K 与 Hit@K 等价。" if single_gold else "Gold page count varies; Recall@K and Hit@K differ."
        ),
        "page": {key: round(value, 4) for key, value in page.items()},
        "chunk": {key: (round(value, 4) if isinstance(value, float) else value) for key, value in chunk.items()},
        "page_chunk": page_chunk,
        "span_chunk": span_chunk,
        "span_qrel": {
            "mapped": mapped,
            "unmapped": unmapped,
        },
        "bootstrap_95": {
            "page_mrr": bootstrap_mean_ci([float(item["page"]["mrr"]) for item in cases]),
            "page_hit_at_5": bootstrap_mean_ci([float(item["page"]["hit_at"]["5"]) for item in cases]),
            "page_recall_at_5": bootstrap_mean_ci(
                [float(item["page"]["recall_at"]["5"]) for item in cases]
            ),
            "page_ndcg_at_10": bootstrap_mean_ci(
                [float(item["page"]["ndcg_at"]["10"]) for item in cases]
            ),
            "chunk_mrr": bootstrap_mean_ci([float(item["chunk"]["mrr"]) for item in cases]),
            "chunk_recall_at_10": bootstrap_mean_ci(
                [float(item["chunk"]["recall_at"]["10"]) for item in cases]
            ),
        },
        "bootstrap_note": (
            "Independent CIs describe one system's uncertainty. They are not a "
            "significance test against another system on the same queries."
        ),
    }


def breakdowns(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "question_type": {},
        "reasoning_type": {},
        "company": {},
        "document_type": {},
        "evidence_pages": {},
        "numeric_extraction": {},
        "multi_step_calculation": {},
        "period_disambiguation": {},
        "negation": {},
        "cross_document": {},
    }
    for case in cases:
        labels = case.get("labels") or {}
        for key in groups:
            raw = labels.get(key, False)
            label = str(raw)
            groups[key].setdefault(label, []).append(case)
    return {
        key: {
            label: {
                "cases": len(items),
                "page_hit_at_5": round(
                    mean(float(item["page"]["hit_at"]["5"]) for item in items), 4
                ),
                "page_mrr": round(mean(float(item["page"]["mrr"]) for item in items), 4),
            }
            for label, items in sorted(bucket.items())
        }
        for key, bucket in groups.items()
    }


def relative_change(baseline: float, current: float) -> float:
    if abs(baseline) < 1e-12:
        return 0.0 if abs(current) < 1e-12 else 1.0
    return (current - baseline) / baseline


def compare_modes(per_case_by_mode: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    modes = list(per_case_by_mode)
    if len(modes) < 2:
        return {"modes": modes, "movements": [], "status": "NOT_RUN"}
    by_id: dict[str, dict[str, dict[str, Any]]] = {}
    for mode, rows in per_case_by_mode.items():
        for row in rows:
            by_id.setdefault(str(row["case_id"]), {})[mode] = row
    movements: list[dict[str, Any]] = []
    improved = 0
    degraded = 0
    never_retrieved = 0
    baseline = modes[0]
    for case_id, mode_rows in sorted(by_id.items()):
        ranks = {
            mode: int(mode_rows[mode]["page"]["first_relevant_rank"] or 0)
            if mode in mode_rows
            else 0
            for mode in modes
        }
        hits = {
            mode: float(mode_rows[mode]["page"]["hit_at"]["5"]) if mode in mode_rows else 0.0
            for mode in modes
        }
        if all(value <= 0 for value in ranks.values()):
            never_retrieved += 1
            kind = "never_retrieved"
        elif ranks.get(modes[-1], 0) and (
            not ranks.get(baseline, 0) or ranks[modes[-1]] < ranks[baseline]
        ):
            improved += 1
            kind = "improved"
        elif ranks.get(baseline, 0) and (
            not ranks.get(modes[-1], 0) or ranks[modes[-1]] > ranks[baseline]
        ):
            degraded += 1
            kind = "degraded"
        else:
            kind = "unchanged"
        movements.append(
            {
                "case_id": case_id,
                "kind": kind,
                "ranks": ranks,
                "hit_at_5": hits,
            }
        )
    aggregates = {mode: aggregate_case_metrics(rows) for mode, rows in per_case_by_mode.items()}
    relative = {}
    paired: dict[str, Any] = {}
    if baseline in aggregates:
        for mode, summary in aggregates.items():
            if mode == baseline:
                continue
            relative[mode] = {
                "page_mrr": round(
                    relative_change(
                        float(aggregates[baseline]["page"]["mrr"]),
                        float(summary["page"]["mrr"]),
                    ),
                    4,
                ),
                "page_hit_at_5": round(
                    relative_change(
                        float(aggregates[baseline]["page"]["hit_at_5"]),
                        float(summary["page"]["hit_at_5"]),
                    ),
                    4,
                ),
            }
            paired[f"{baseline}_vs_{mode}"] = compare_paired_systems(
                per_case_by_mode[baseline],
                per_case_by_mode[mode],
                baseline_name=baseline,
                candidate_name=mode,
            )
    named_pairs = []
    if "dense" in per_case_by_mode and "hybrid" in per_case_by_mode:
        named_pairs.append(("dense", "hybrid"))
    if "hybrid" in per_case_by_mode and "hybrid-qwen3" in per_case_by_mode:
        named_pairs.append(("hybrid", "hybrid-qwen3"))
    if "dense" in per_case_by_mode and "hybrid-qwen3" in per_case_by_mode:
        named_pairs.append(("dense", "hybrid-qwen3"))
    for left, right in named_pairs:
        key = f"{left}_vs_{right}"
        if key not in paired:
            paired[key] = compare_paired_systems(
                per_case_by_mode[left],
                per_case_by_mode[right],
                baseline_name=left,
                candidate_name=right,
            )
    return {
        "modes": modes,
        "baseline": baseline,
        "improved": improved,
        "degraded": degraded,
        "never_retrieved": never_retrieved,
        "aggregates": aggregates,
        "relative_change_vs_baseline": relative,
        "paired": paired,
        "movements": movements,
    }


def render_markdown(report: dict[str, Any]) -> str:
    env = report.get("environment") or {}
    summary = report.get("summary") or {}
    page = summary.get("page") or {}
    chunk = summary.get("chunk") or {}
    system = report.get("system") or {}
    lines = [
        "# FinanceBench retrieval evaluation",
        "",
        f"- schema: `{report.get('schema_version')}`",
        f"- mode: `{env.get('mode')}`",
        f"- split: `{env.get('split')}`",
        f"- remote calls: `{env.get('remote_calls_enabled')}`",
        f"- commit: `{env.get('lumenfin_commit')}` ({env.get('worktree_status')})",
        f"- status: `{report.get('status', 'recorded')}`",
        "",
        "## Page-level metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, label in (
        ("hit_at_1", "Hit@1"),
        ("hit_at_3", "Hit@3"),
        ("hit_at_5", "Hit@5"),
        ("hit_at_10", "Hit@10"),
        ("recall_at_1", "Recall@1"),
        ("recall_at_3", "Recall@3"),
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
        ("mrr", "MRR"),
        ("ndcg_at_5", "nDCG@5"),
        ("ndcg_at_10", "nDCG@10"),
    ):
        lines.append(f"| {label} | {page.get(key, 'NOT_RUN')} |")
    lines.extend(
        [
            "",
            str(summary.get("recall_equals_hit_note") or ""),
            "",
            "## Chunk-level metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for key, label in (
        ("hit_at_5", "Hit@5"),
        ("hit_at_10", "Hit@10"),
        ("hit_at_20", "Hit@20"),
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
        ("recall_at_20", "Recall@20"),
        ("mrr", "MRR"),
        ("ndcg_at_10", "nDCG@10"),
    ):
        lines.append(f"| {label} | {chunk.get(key, 'NOT_RUN')} |")
    ci = (summary.get("bootstrap_95") or {}).get("page_mrr") or {}
    lines.extend(
        [
            "",
            "## Bootstrap 95% CI (page MRR)",
            "",
            f"- mean: {ci.get('mean', 'NOT_RUN')}",
            f"- low: {ci.get('ci95_low', 'NOT_RUN')}",
            f"- high: {ci.get('ci95_high', 'NOT_RUN')}",
            "",
            "## System",
            "",
            f"- indexing_ms: {system.get('indexing_ms', 'NOT_RUN')}",
            f"- query_p50_ms: {system.get('query_p50_ms', 'NOT_RUN')}",
            f"- query_p95_ms: {system.get('query_p95_ms', 'NOT_RUN')}",
            f"- embedding_calls: {system.get('embedding_calls', 'NOT_RUN')}",
            f"- rerank_calls: {system.get('rerank_calls', 'NOT_RUN')}",
            f"- rerank_fallback_rate: {system.get('rerank_fallback_rate', 'NOT_RUN')}",
            "",
            "Unrun confirmation-50 numbers are marked `NOT_RUN`.",
            "test-100 is an exposed exploratory baseline, not a unseen held-out.",
            "Independent CIs are not a paired significance test.",
            "Synthetic 10-case Qwen3 gates are not FinanceBench accuracy.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def not_run_report(*, reason: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "NOT_RUN",
        "reason": reason,
        "summary": {"cases": 0, "page": {}, "chunk": {}},
    }
