from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class HttpResult:
    status_code: int
    body: Any
    headers: dict[str, str]
    url: str
    elapsed_ms: float


def request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> HttpResult:
    import time

    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            body = json.loads(raw) if raw.strip() else None
            return HttpResult(
                status_code=int(response.status),
                body=body,
                headers={k.lower(): v for k, v in response.headers.items()},
                url=url,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            body = json.loads(raw) if raw.strip() else {"detail": str(exc)}
        except json.JSONDecodeError:
            body = {"detail": raw or str(exc)}
        return HttpResult(
            status_code=int(exc.code),
            body=body,
            headers={k.lower(): v for k, v in exc.headers.items()} if exc.headers else {},
            url=url,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def multipart_index(
    url: str,
    *,
    files: list[tuple[str, bytes]],
    tenant_id: str,
    async_mode: bool = True,
    timeout: float = 120.0,
) -> HttpResult:
    import time
    import uuid

    boundary = f"----lumenfin{uuid.uuid4().hex}"
    body = bytearray()

    def add_field(name: str, value: str) -> None:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    add_field("tenant_id", tenant_id)
    add_field("async_mode", "true" if async_mode else "false")
    for filename, content in files:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            (
                f'Content-Disposition: form-data; name="files"; filename="{filename}"\r\n'
                "Content-Type: text/markdown\r\n\r\n"
            ).encode()
        )
        body.extend(content)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return HttpResult(
                status_code=int(response.status),
                body=json.loads(raw) if raw.strip() else None,
                headers={k.lower(): v for k, v in response.headers.items()},
                url=url,
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8")
        try:
            parsed = json.loads(raw) if raw.strip() else {"detail": str(exc)}
        except json.JSONDecodeError:
            parsed = {"detail": raw or str(exc)}
        return HttpResult(
            status_code=int(exc.code),
            body=parsed,
            headers={k.lower(): v for k, v in exc.headers.items()} if exc.headers else {},
            url=url,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def analyze(url: str, query: str, thread_id: str, *, export_artifacts: bool = False) -> HttpResult:
    return request_json(
        "POST",
        f"{url.rstrip('/')}/api/v1/analyze",
        payload={
            "query": query,
            "thread_id": thread_id,
            "export_artifacts": export_artifacts,
            "include_state": True,
        },
    )
