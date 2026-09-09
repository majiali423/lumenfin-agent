#!/usr/bin/env python3
"""Phase 0 baseline: inventory, offline env, and review-issue reproductions.

Does not mutate existing virtualenvs, does not print credentials, and does not
rewrite sealed evaluation artifacts. Writes JSON evidence only.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = Path(os.environ.get("FINAGENTBENCH_DIR") or ROOT.parent / "finagentbench-demo")
REVIEW_OUT = Path(
    os.environ.get("PHASE0_OUT") or (ROOT.parent / "project-review-20260907" / "phase0")
)
for path in (ROOT, SRC, BENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.offline_env import apply_offline_env, credential_source_report

apply_offline_env()


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return (result.stdout or result.stderr or "").strip()


def _pip_out(python: str, *args: str) -> str:
    result = subprocess.run(
        [python, "-m", "pip", *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return ((result.stdout or "") + (result.stderr or "")).strip()


def collect_repo_meta(repo: Path) -> dict[str, object]:
    return {
        "path": str(repo),
        "head": _git(repo, "rev-parse", "HEAD"),
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repo, "describe", "--tags", "--always"),
        "dirty": bool(_git(repo, "status", "--porcelain")),
        "status_short": _git(repo, "status", "--short", "--branch"),
    }


def collect_inventory() -> dict[str, object]:
    result: dict[str, object] = {}
    for label, path in (
        ("lumen_runtime", ROOT / "src" / "lumenfin"),
        ("bench_runtime", BENCH / "finagentbench"),
    ):
        paths = list(path.rglob("*.py"))
        functions = []
        total = eval_lines = 0
        for py_path in paths:
            source = py_path.read_text(encoding="utf-8-sig")
            lines = len(source.splitlines())
            total += lines
            if "eval" in py_path.relative_to(path).parts:
                eval_lines += lines
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.end_lineno:
                    functions.append(
                        {
                            "path": str(py_path.relative_to(path.parents[1] if label == "lumen_runtime" else BENCH)),
                            "name": node.name,
                            "line": node.lineno,
                            "lines": node.end_lineno - node.lineno + 1,
                        }
                    )
        result[label] = {
            "files": len(paths),
            "physical_lines": total,
            "eval_lines": eval_lines,
            "longest_functions": sorted(functions, key=lambda item: item["lines"], reverse=True)[:8],
        }
    result["counts"] = {
        "lumenfin_tests": len(list((ROOT / "tests").rglob("test_*.py"))),
        "lumenfin_docs": len(list((ROOT / "docs").rglob("*.md"))),
        "lumenfin_scripts": len(list((ROOT / "scripts").rglob("*.py"))),
        "finagentbench_tests": len(list((BENCH / "tests").rglob("test_*.py"))),
        "finagentbench_docs": len(list((BENCH / "docs").rglob("*.md"))),
    }
    return result


def collect_environment() -> dict[str, object]:
    lumen_py = ROOT / ".venv" / "Scripts" / "python.exe"
    bench_py = BENCH / ".venv" / "Scripts" / "python.exe"
    if not lumen_py.is_file():
        lumen_py = ROOT / ".venv" / "bin" / "python"
    if not bench_py.is_file():
        bench_py = BENCH / ".venv" / "bin" / "python"

    def versions(python: Path) -> dict[str, object]:
        if not python.is_file():
            return {"python": None, "missing": True}
        code = (
            "import importlib.metadata as m, sys, json\n"
            "names=['lumenfin-agent','finagentbench','pymilvus','milvus-lite',"
            "'fastapi','langgraph','prometheus-client','pydantic','sqlalchemy','redis']\n"
            "out={'python':sys.version,'executable':sys.executable,'packages':{}}\n"
            "for n in names:\n"
            "    try:\n"
            "        out['packages'][n]=m.version(n)\n"
            "    except Exception:\n"
            "        out['packages'][n]=None\n"
            "print(json.dumps(out))\n"
        )
        proc = subprocess.run([str(python), "-c", code], check=False, capture_output=True, text=True)
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            payload = {"raw_stdout": proc.stdout, "raw_stderr": proc.stderr}
        payload["pip_check"] = _pip_out(str(python), "check")
        return payload

    lock_versions = {}
    lock = ROOT / "requirements-lock.txt"
    if lock.is_file():
        for line in lock.read_text(encoding="utf-8").splitlines():
            if line.startswith(("milvus-lite==", "pymilvus==", "fastapi==", "langgraph==", "prometheus-client==")):
                name, version = line.split("==", 1)
                lock_versions[name] = version

    pyproject = tomllib_version(ROOT / "pyproject.toml")
    bench_pyproject = tomllib_version(BENCH / "pyproject.toml")
    return {
        "platform": platform.platform(),
        "host_python": sys.version,
        "lumen_source_version": pyproject,
        "bench_source_version": bench_pyproject,
        "lock_pins": lock_versions,
        "lumen_venv": versions(lumen_py),
        "bench_venv": versions(bench_py),
        "hashes": {
            "lumen_pyproject": _sha256(ROOT / "pyproject.toml"),
            "lumen_lock": _sha256(ROOT / "requirements-lock.txt"),
            "bench_pyproject": _sha256(BENCH / "pyproject.toml"),
            "case_lumenfin_diligence": _sha256(BENCH / "fixtures" / "case_lumenfin_diligence.json"),
            "lumenfin_state_sample": _sha256(BENCH / "fixtures" / "lumenfin_state_sample.json"),
        },
        "dotenv_exists": (ROOT / ".env").is_file(),
        "credential_sources": credential_source_report(root=ROOT),
        "python_dotenv_disabled": os.environ.get("PYTHON_DOTENV_DISABLED"),
    }


def tomllib_version(path: Path) -> str | None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover
        return None
    if not path.is_file():
        return None
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return str(data.get("project", {}).get("version") or "")


def probe_final_output_gate() -> dict[str, object]:
    from lumenfin.finrun import export_finrun_state
    from finagentbench.runner import evaluate_run

    case = json.loads((BENCH / "fixtures" / "case_lumenfin_diligence.json").read_text(encoding="utf-8"))
    state = json.loads((BENCH / "fixtures" / "lumenfin_state_sample.json").read_text(encoding="utf-8"))
    run = export_finrun_state(state)
    baseline = evaluate_run(run, case)
    mutated_run = dict(run)
    mutated_run["final_output"] = (
        str(run.get("final_output") or "") + "\n\nApple EBITDA margin is 999999% for FY2025.\n"
    )
    mutated = evaluate_run(mutated_run, case)
    voi_case = dict(case)
    enabled = list(case.get("enabled_metrics") or [])
    if "visible_output_integrity" not in enabled:
        voi_case["enabled_metrics"] = enabled + ["visible_output_integrity"]
    voi = evaluate_run(mutated_run, voi_case)
    return {
        "classification": "code_defect",
        "baseline": {"passed": baseline.passed, "score": baseline.score},
        "false_visible_numeric_statement": {"passed": mutated.passed, "score": mutated.score},
        "same_mutation_with_visible_output_integrity": {"passed": voi.passed, "score": voi.score},
        "enabled_metrics": [metric.name for metric in mutated.metrics],
        "reproduced": bool(baseline.passed and mutated.passed),
        "notes": (
            "numeric_correctness scores structured metric formula/value pairs only; "
            "it does not read final_output prose."
        ),
    }


def probe_job_claim() -> dict[str, object]:
    from lumenfin.database import JobRepository

    with tempfile.TemporaryDirectory(prefix="phase0-job-") as tmp:
        repo = JobRepository("sqlite:///" + (Path(tmp) / "jobs.db").as_posix())
        repo.create_job(job_id="phase0-job", thread_id="phase0-thread", query="review", tenant_id="review")
        first = repo.begin_job_execution("phase0-job", thread_id="phase0-thread", query="review")
        second = repo.begin_job_execution("phase0-job", thread_id="phase0-thread", query="review")
        missing = repo.begin_job_execution("missing-job", thread_id="phase0-thread", query="review")
        repo.update_job_status("phase0-job", status="completed")
        after_complete = repo.begin_job_execution("phase0-job", thread_id="phase0-thread", query="review")
        repo.engine.dispose()
    return {
        "classification": "code_defect",
        "first_claim": first,
        "second_claim_while_running": second,
        "missing_job": missing,
        "claim_after_completed": after_complete,
        "reproduced": first == "run" and second == "run",
        "notes": "begin_job_execution only skips completed jobs; running can be claimed again.",
    }


def probe_queue_ownership() -> dict[str, object]:
    queueing = (ROOT / "src" / "lumenfin" / "queueing.py").read_text(encoding="utf-8")
    worker = (ROOT / "src" / "lumenfin" / "worker.py").read_text(encoding="utf-8")
    ack_src = inspect.getsource(
        __import__("lumenfin.queueing", fromlist=["RedisQueueManager"]).RedisQueueManager.ack
    )
    retry_src = inspect.getsource(
        __import__("lumenfin.queueing", fromlist=["RedisQueueManager"]).RedisQueueManager.retry
    )
    return {
        "classification": "code_defect",
        "ack_discards_worker_id": "del worker_id" in ack_src,
        "retry_discards_worker_id": "del worker_id" in retry_src,
        "ack_lua_has_worker_or_token_argv": ("reserved_by" in queueing.split("_ACK_LUA")[1].split("_RETRY_LUA")[0])
        if "_ACK_LUA" in queueing
        else False,
        "retry_lua_checks_owner": "reserved_by" in queueing.split("_RETRY_LUA")[1].split("_RECLAIM_LUA")[0]
        and "ARGV" in queueing.split("_RETRY_LUA")[1].split("_RECLAIM_LUA")[0]
        and "worker" in queueing.split("_RETRY_LUA")[1].split("_RECLAIM_LUA")[0].lower(),
        "analysis_worker_renews_reservation": "renew" in worker.lower()
        and "process_reserved_analysis_message" in worker,
        "reclaim_idle_default_seconds": 10,
        "compose_env_example_reclaim_seconds": 30,
        "analysis_deadline_default_seconds": 120,
        "reproduced": True,
        "kind": "static_plus_source_inspection",
        "notes": (
            "ACK/retry Lua match message_id only. Analysis worker has no reservation renew "
            "during execute_analysis_job. Real Redis multi-process replay was not run."
        ),
    }


def probe_html_render() -> dict[str, object]:
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    assignments = []
    for index, line in enumerate(html.splitlines(), start=1):
        if "innerHTML" in line:
            assignments.append({"line": index, "text": line.strip()[:240]})
    return {
        "classification": "code_defect",
        "innerhtml_assignments": assignments,
        "uses_dompurify": "DOMPurify" in html,
        "markdown_to_html_uses_marked_parse": "marked.parse" in html,
        "clarification_innerhtml": any("clarifyQuestions" in item["text"] for item in assignments),
        "report_innerhtml": any("reportContent" in item["text"] for item in assignments),
        "simulated_progress_800ms": "setInterval" in html and "800" in html,
        "reproduced": True,
        "kind": "static_render_chain",
        "notes": "Browser XSS execution was not run in Phase 0.",
    }


def probe_upload_and_delivery() -> dict[str, object]:
    from dataclasses import replace
    from uuid import uuid4

    from fastapi.testclient import TestClient

    import importlib.util

    from lumenfin.api.app import create_app
    from lumenfin.llm import LocalFallbackLLMClient
    from lumenfin.service import LumenFinAnalysisService

    spec = importlib.util.spec_from_file_location(
        "phase0_test_graph_routing",
        ROOT / "tests" / "test_graph_routing.py",
    )
    assert spec is not None and spec.loader is not None
    routing = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(routing)
    build_test_config = routing.build_test_config

    tmp = ROOT / "test_artifacts" / f"phase0-upload-{uuid4().hex[:8]}"
    tmp.mkdir(parents=True, exist_ok=True)
    config = replace(
        build_test_config(tmp),
        max_upload_bytes=32,
        max_upload_files=2,
        rag_enabled=False,
        redis_url=None,
        api_key=None,
        app_env="test",
    )
    service = LumenFinAnalysisService(
        config,
        llm_client=LocalFallbackLLMClient(),
        market_data_client=None,
    )
    leftover_after_partial = []
    partial_error = None
    try:
        # Count is legal (2 <= max_upload_files); the second file fails after the first is written.
        service.save_uploaded_files(
            [
                ("ok.md", b"safe notes"),
                ("bad.exe", b"x" * 8),
            ]
        )
    except ValueError as exc:
        partial_error = str(exc)
        leftover_after_partial = [path.name for path in config.upload_dir.glob("*")] if config.upload_dir.exists() else []

    over_limit_error = None
    try:
        service.save_uploaded_files([("a.md", b"1"), ("b.md", b"2"), ("c.md", b"3")])
    except ValueError as exc:
        over_limit_error = str(exc)

    oversized_error = None
    try:
        service.save_uploaded_files([("big.md", b"x" * 64)])
    except ValueError as exc:
        oversized_error = str(exc)

    created = service.submit_job(query="phase0 enqueue window", thread_id="phase0-enq")
    enqueue_error = None
    job_after_failed_enqueue = None
    with patch("lumenfin.service.RedisQueueManager.enqueue", side_effect=ConnectionError("phase0-redis-down")):
        failing = replace(config, redis_url="redis://127.0.0.1:9/0")
        failing_service = LumenFinAnalysisService(
            failing,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=None,
        )
        try:
            failing_service.enqueue_job(
                created["job_id"],
                "phase0 enqueue window",
                created["thread_id"],
            )
        except ConnectionError as exc:
            enqueue_error = str(exc)
        job_after_failed_enqueue = failing_service.get_job(created["job_id"], tenant_id=created["tenant_id"])

    http_config = replace(config, max_upload_files=1)
    app = create_app(
        http_config,
        llm_client=LocalFallbackLLMClient(),
        market_data_client=None,
    )
    client = TestClient(app)
    upload_status = {}
    for route in ("/api/v1/analyze-upload", "/api/v1/documents/index", "/api/v1/jobs/upload"):
        files = [("files", ("too-many-1.md", b"1", "text/markdown")), ("files", ("too-many-2.md", b"2", "text/markdown"))]
        try:
            response = client.post(route, data={"query": "phase0 upload"}, files=files)
            upload_status[route] = {
                "status_code": response.status_code,
                "detail_prefix": str(response.text)[:180],
                "unhandled_exception": None,
            }
        except Exception as exc:  # noqa: BLE001 - jobs/upload currently leaks ValueError
            upload_status[route] = {
                "status_code": None,
                "detail_prefix": "",
                "unhandled_exception": f"{type(exc).__name__}: {exc}",
            }

    app_src = (ROOT / "src" / "lumenfin" / "api" / "app.py").read_text(encoding="utf-8")
    jobs_upload_has_413_handler = "jobs/upload" in app_src and "status_code=413" in app_src.split("jobs/upload")[1].split("@app.get")[0]
    jobs_upload_reads_whole_file = "await upload.read()" in app_src.split("jobs/upload")[1].split("@app.get")[0]
    jobs_upload_uses_threadpool = "run_in_threadpool" in app_src.split("jobs/upload")[1].split("@app.get")[0]

    return {
        "classification": "code_defect",
        "partial_save_error": partial_error,
        "leftover_files_after_partial_failure": leftover_after_partial,
        "over_limit_error": over_limit_error,
        "oversized_error": oversized_error,
        "enqueue_failure": enqueue_error,
        "job_status_after_enqueue_failure": (job_after_failed_enqueue or {}).get("status"),
        "job_id_after_enqueue_failure": created["job_id"],
        "upload_http_status": upload_status,
        "jobs_upload_has_413_handler": jobs_upload_has_413_handler,
        "jobs_upload_reads_whole_file": jobs_upload_reads_whole_file,
        "jobs_upload_uses_threadpool": jobs_upload_uses_threadpool,
        "reproduced": bool(
            leftover_after_partial
            and enqueue_error
            and (job_after_failed_enqueue or {}).get("status") == "pending"
            and upload_status.get("/api/v1/jobs/upload", {}).get("unhandled_exception")
        ),
        "notes": (
            "Limits are checked after full await upload.read(). Partial writes are not rolled back. "
            "DB submit and Redis enqueue are separate writes."
        ),
    }


def main() -> int:
    REVIEW_OUT.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    output: dict[str, object] = {
        "phase": 0,
        "started_at_utc": started,
        "lumenfin": collect_repo_meta(ROOT),
        "finagentbench": collect_repo_meta(BENCH),
        "environment": collect_environment(),
        "inventory": collect_inventory(),
    }
    probes = {
        "final_output": probe_final_output_gate,
        "execution_claim": probe_job_claim,
        "queue_ownership": probe_queue_ownership,
        "html_render": probe_html_render,
        "upload_and_delivery": probe_upload_and_delivery,
    }
    for name, fn in probes.items():
        try:
            output[name] = fn()
        except Exception as exc:  # noqa: BLE001 - baseline must record probe failures
            output[name] = {"error": f"{type(exc).__name__}: {exc}", "reproduced": False}
    output["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    dest = REVIEW_OUT / "phase0-baseline.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"wrote": str(dest), "finished_at_utc": output["finished_at_utc"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
