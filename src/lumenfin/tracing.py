"""Optional LangSmith tracing. Fail-open; never changes product results."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)

_trace_enabled: ContextVar[bool] = ContextVar("lumenfin_trace_enabled", default=False)
_trace_meta: ContextVar[dict[str, Any]] = ContextVar("lumenfin_trace_meta", default={})
_local_events: ContextVar[list[dict[str, Any]]] = ContextVar("lumenfin_trace_events", default=[])
_parent_id: ContextVar[str | None] = ContextVar("lumenfin_trace_parent", default=None)
_eval_case_meta: ContextVar[dict[str, Any]] = ContextVar("lumenfin_eval_case", default={})
_remote_parent: ContextVar[dict[str, Any] | None] = ContextVar("lumenfin_remote_parent", default=None)
_completed_trace: ContextVar[dict[str, Any]] = ContextVar("lumenfin_completed_trace", default={})


def tracing_requested() -> bool:
    flag = (os.getenv("MAS_LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2") or "").strip().lower()
    return flag in {"1", "true", "yes"}


def langsmith_api_key() -> str:
    return (os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY") or "").strip()


def tracing_configured() -> bool:
    return tracing_requested() and bool(langsmith_api_key())


def langsmith_project() -> str:
    return (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "lumenfin-document-task-eval"
    ).strip()


def enable_live_eval_tracing() -> None:
    """Process-local: turn tracing on when a key is present so eval cases are posted."""
    if langsmith_api_key():
        os.environ["MAS_LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGSMITH_PROJECT", langsmith_project())


@contextmanager
def eval_case_context(metadata: dict[str, Any]) -> Iterator[None]:
    token = _eval_case_meta.set(_safe_meta(dict(metadata or {})))
    try:
        yield
    finally:
        _eval_case_meta.reset(token)


def completed_analysis_trace() -> dict[str, Any]:
    return dict(_completed_trace.get() or {})


def _local_trace_dir() -> Path:
    raw = (os.getenv("MAS_TRACE_DIR") or "").strip()
    if raw:
        return Path(raw)
    return Path("outputs") / "traces"


def _safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    blocked = ("api_key", "authorization", "password", "secret", "token")
    out: dict[str, Any] = {}
    for key, value in meta.items():
        lowered = str(key).lower()
        if any(item in lowered for item in blocked):
            continue
        out[key] = value
    return out


def _truncate(text: str, limit: int = 240) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def current_trace_events() -> list[dict[str, Any]]:
    return list(_local_events.get() or [])


@contextmanager
def analysis_trace(
    *,
    name: str = "lumenfin.analyze",
    metadata: dict[str, Any] | None = None,
    inputs: dict[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Parent span for one analysis. Product exceptions propagate; tracer errors do not."""
    run_id = uuid4().hex
    meta = _safe_meta({**(_eval_case_meta.get() or {}), **(metadata or {})})
    events: list[dict[str, Any]] = []
    token_enabled = _trace_enabled.set(True)
    token_meta = _trace_meta.set(meta)
    token_events = _local_events.set(events)
    token_parent = _parent_id.set(run_id)
    token_remote = _remote_parent.set(None)
    started = time.perf_counter()
    record: dict[str, Any] = {
        "run_id": run_id,
        "name": name,
        "ok": True,
        "remote": "not_attempted",
        "events": events,
        "metadata": meta,
        "langsmith_run_id": None,
        "langsmith_url": None,
    }
    try:
        if tracing_configured():
            try:
                remote = _start_remote_parent(name=name, meta=meta, inputs=inputs or {})
                if remote:
                    _remote_parent.reset(token_remote)
                    token_remote = _remote_parent.set(remote)
                    record["remote"] = "entered"
                    record["langsmith_run_id"] = str(remote["run_id"])
                    record["langsmith_url"] = remote.get("url")
            except Exception as exc:
                record["remote"] = f"error:{type(exc).__name__}"
                logger.warning("LangSmith tracing failed open: %s", type(exc).__name__)
        yield record
    except Exception:
        record["ok"] = False
        raise
    finally:
        record["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        try:
            if tracing_requested() or os.getenv("MAS_TRACE_DIR", "").strip():
                _persist_local(record, inputs=inputs)
        except Exception:
            logger.warning("local trace persist failed", exc_info=True)
        try:
            remote = _remote_parent.get()
            if remote is not None:
                _finish_remote_parent(remote, record)
                if record.get("remote") == "entered":
                    record["remote"] = "completed"
        except Exception as exc:
            record["remote"] = f"error:{type(exc).__name__}"
            logger.warning("LangSmith tracing failed open: %s", type(exc).__name__)
        _completed_trace.set(
            {
                "run_id": record["run_id"],
                "langsmith_run_id": record.get("langsmith_run_id"),
                "langsmith_url": record.get("langsmith_url"),
                "remote": record.get("remote"),
                "ok": record.get("ok"),
                "events": list(record.get("events") or []),
            }
        )
        _trace_enabled.reset(token_enabled)
        _trace_meta.reset(token_meta)
        _local_events.reset(token_events)
        _parent_id.reset(token_parent)
        _remote_parent.reset(token_remote)


def _start_remote_parent(*, name: str, meta: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any] | None:
    from langsmith import Client

    client = Client()
    run_uuid = uuid4()
    project = langsmith_project()
    tags = [
        "lumenfin",
        "document-task-eval",
        str(meta.get("system") or "product"),
        str(meta.get("task_id") or meta.get("case_id") or ""),
    ]
    client.create_run(
        name=name,
        run_type="chain",
        inputs={
            "query": _truncate(str(inputs.get("query") or ""), 400),
            "task_id": meta.get("task_id") or meta.get("case_id"),
            "system": meta.get("system"),
            "model": meta.get("model"),
        },
        extra={"metadata": meta},
        tags=[item for item in tags if item],
        project_name=project,
        id=run_uuid,
    )
    url = None
    try:
        fetched = client.read_run(run_uuid)
        url = getattr(fetched, "url", None)
    except Exception:
        url = f"https://smith.langchain.com/runs/{run_uuid}"
    return {"client": client, "run_id": run_uuid, "url": url, "project": project}


def _finish_remote_parent(remote: dict[str, Any], record: dict[str, Any]) -> None:
    client = remote["client"]
    outputs = dict(record.get("outputs") or {})
    outputs.setdefault("ok", record.get("ok"))
    outputs.setdefault("duration_ms", record.get("duration_ms"))
    usage_events = [item for item in (record.get("events") or []) if item.get("name") == "llm.chat"]
    if usage_events:
        outputs["model_calls"] = [
            {
                "model": item.get("model"),
                "prompt_tokens": item.get("prompt_tokens"),
                "completion_tokens": item.get("completion_tokens"),
                "attempts": item.get("attempts"),
                "ok": item.get("ok"),
            }
            for item in usage_events
        ]
    client.update_run(
        remote["run_id"],
        outputs=outputs,
        extra={"metadata": record.get("metadata") or {}},
        end_time=datetime.now(timezone.utc),
        error=None if record.get("ok") else "product_error",
    )


def _remote_tracing_context(meta: dict[str, Any]):
    try:
        from langsmith.run_helpers import tracing_context
    except Exception:
        return None
    return tracing_context(
        project_name=langsmith_project(),
        enabled=True,
        metadata=meta,
        tags=["lumenfin", str(meta.get("system") or "product")],
    )


def probe_langsmith_remote() -> dict[str, Any]:
    """Create and read a real run. Key presence alone is not success."""
    payload: dict[str, Any] = {
        "ok": False,
        "requested": tracing_requested(),
        "key_present": bool(langsmith_api_key()),
        "configured": tracing_configured(),
        "project": langsmith_project(),
    }
    if not payload["key_present"]:
        payload["reason"] = "key_missing"
        return payload
    try:
        from langsmith import Client

        client = Client()
        run_id = uuid4()
        client.create_run(
            name="lumenfin.eval.langsmith_probe",
            run_type="chain",
            inputs={"probe": True, "task_id": "langsmith-probe", "system": "connectivity"},
            extra={"metadata": {"task_id": "langsmith-probe", "system": "connectivity"}},
            tags=["lumenfin", "langsmith-probe"],
            project_name=payload["project"],
            id=run_id,
        )
        payload["run_id"] = str(run_id)
        patched = False
        try:
            client.update_run(
                run_id,
                outputs={"verified": True},
                extra={"metadata": {"task_id": "langsmith-probe", "system": "connectivity"}},
                end_time=datetime.now(timezone.utc),
            )
            patched = True
        except Exception as exc:
            payload["patch_error"] = type(exc).__name__
        fetched = None
        url = None
        for delay in (0.0, 0.5, 1.0, 2.0):
            if delay:
                time.sleep(delay)
            try:
                fetched = client.read_run(run_id)
                url = getattr(fetched, "url", None)
                if url:
                    break
            except Exception as exc:
                payload["read_error"] = type(exc).__name__
        name_match = bool(fetched) and str(getattr(fetched, "name", "") or "") == "lumenfin.eval.langsmith_probe"
        id_match = bool(fetched) and str(getattr(fetched, "id", "")) == str(run_id)
        if not url:
            url = f"https://smith.langchain.com/runs/{run_id}"
        payload.update(
            {
                "ok": patched or bool(fetched),
                "url": url,
                "name_match": name_match,
                "id_match": id_match,
                "created": True,
                "patched": patched,
                "read": bool(fetched),
            }
        )
        if not payload["ok"]:
            payload["reason"] = "create_or_patch_failed"
        return payload
    except Exception as exc:
        payload["reason"] = type(exc).__name__
        payload["ok"] = False
        return payload


def patch_remote_run(
    run_id: str | None,
    *,
    outputs: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
) -> bool:
    """Attach gold/scoring without replacing the parent end event."""
    if not run_id or not tracing_configured():
        return False
    try:
        from langsmith import Client

        client = Client()
        parent = UUID(str(run_id))
        child_id = uuid4()
        meta = _safe_meta(dict(extra_metadata or {}))
        body = _safe_meta(outputs)
        project = langsmith_project()
        client.create_run(
            name="lumenfin.eval.gold",
            run_type="chain",
            parent_run_id=parent,
            inputs={
                "task_id": meta.get("task_id") or body.get("task_id"),
                "system": meta.get("system") or body.get("system"),
            },
            extra={"metadata": meta},
            tags=["lumenfin", "eval-gold", str(meta.get("system") or "")],
            project_name=project,
            id=child_id,
        )
        client.update_run(
            child_id,
            outputs=body,
            extra={"metadata": meta},
            end_time=datetime.now(timezone.utc),
        )
        try:
            client.update_run(parent, extra={"metadata": {**meta, "gold_attached": True}})
        except Exception:
            pass
        return True
    except Exception as exc:
        logger.warning("LangSmith score patch failed open: %s", type(exc).__name__)
        return False


@contextmanager
def node_span(name: str, *, extra: dict[str, Any] | None = None) -> Iterator[None]:
    if not _trace_enabled.get():
        yield
        return
    started = time.perf_counter()
    event: dict[str, Any] = {
        "name": name,
        "parent_run_id": _parent_id.get(),
        "ok": True,
        "extra": _safe_meta(dict(extra or {})),
    }
    try:
        yield
    except Exception as exc:
        event["ok"] = False
        event["error_type"] = type(exc).__name__
        event["error"] = _truncate(str(exc), 160)
        raise
    finally:
        event["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        bucket = _local_events.get()
        if bucket is not None:
            bucket.append(event)
        try:
            _emit_remote_child(event)
        except Exception:
            logger.warning("LangSmith child span failed open", exc_info=True)


def record_llm_call(
    *,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    attempts: int,
    error: str | None = None,
) -> None:
    if not _trace_enabled.get():
        return
    event = {
        "name": "llm.chat",
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "attempts": attempts,
        "ok": error is None,
        "error": error,
        "parent_run_id": _parent_id.get(),
    }
    bucket = _local_events.get()
    if bucket is not None:
        bucket.append(event)
    try:
        _emit_remote_child(event)
    except Exception:
        logger.warning("LangSmith llm span failed open", exc_info=True)


def attach_llm_tracing(client: Any) -> Any:
    """Wrap ``chat`` once. Tracing failures never alter or hide LLM errors."""
    if getattr(client, "_lumenfin_chat_traced", False):
        return client
    original = client.chat

    def chat(system_prompt: str, user_prompt: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        error = None
        try:
            return original(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            usage = {}
            since = getattr(client, "usage_since_mark", None)
            if callable(since):
                try:
                    usage = since()
                except Exception:
                    usage = {}
            record_llm_call(
                model=str(getattr(client, "model_name", "unknown")),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                attempts=int(getattr(client, "last_attempts", 1) or 1),
                error=error,
            )

    client.chat = chat  # type: ignore[method-assign]
    client._lumenfin_chat_traced = True
    return client


def _emit_remote_child(event: dict[str, Any]) -> None:
    remote = _remote_parent.get()
    if not remote:
        return
    client = remote["client"]
    child_id = uuid4()
    run_type = "llm" if event.get("name") == "llm.chat" else "chain"
    client.create_run(
        name=str(event.get("name") or "child"),
        run_type=run_type,
        parent_run_id=remote["run_id"],
        inputs={
            "model": event.get("model"),
            "attempts": event.get("attempts"),
        },
        extra={
            "metadata": {
                "prompt_tokens": event.get("prompt_tokens"),
                "completion_tokens": event.get("completion_tokens"),
                "ok": event.get("ok"),
            }
        },
        project_name=remote["project"],
        id=child_id,
    )
    client.update_run(
        child_id,
        outputs={
            "ok": event.get("ok"),
            "error": event.get("error"),
            "prompt_tokens": event.get("prompt_tokens"),
            "completion_tokens": event.get("completion_tokens"),
        },
        end_time=datetime.now(timezone.utc),
        error=None if event.get("ok") else str(event.get("error") or "llm_error"),
    )


def _persist_local(record: dict[str, Any], *, inputs: dict[str, Any] | None) -> None:
    directory = _local_trace_dir()
    directory.mkdir(parents=True, exist_ok=True)
    body = {
        **record,
        "inputs": {
            "query": _truncate(str((inputs or {}).get("query") or ""), 200),
            "document_ids": list((inputs or {}).get("document_ids") or [])[:32],
            "job_id": (inputs or {}).get("job_id"),
            "thread_id": (inputs or {}).get("thread_id"),
        },
        "langsmith_url": record.get("langsmith_url"),
        "langsmith_run_id": record.get("langsmith_run_id"),
        "note": "Document full text is not stored. Public eval fixtures are identified by hash only.",
    }
    path = directory / f"{record['run_id']}.json"
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def dump_failed_trace(exc: BaseException) -> None:
    """Keep traceback locally; never used to swallow the original exception."""
    try:
        directory = _local_trace_dir() / "errors"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{uuid4().hex}.txt").write_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            encoding="utf-8",
        )
    except Exception:
        logger.warning("failed to dump trace error file", exc_info=True)
