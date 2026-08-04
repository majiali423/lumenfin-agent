from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .settings import COMPOSE_FILE, ENV_FILE, IntegrationSettings


class DockerUnavailable(RuntimeError):
    pass


def _require_docker() -> None:
    if shutil.which("docker") is None:
        raise DockerUnavailable("docker executable not found on PATH")
    probe = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise DockerUnavailable(probe.stderr.strip() or "docker info failed")


def compose_cmd(settings: IntegrationSettings, *args: str) -> list[str]:
    cmd = [
        "docker",
        "compose",
        "-f",
        str(COMPOSE_FILE),
        "--env-file",
        str(ENV_FILE),
        "-p",
        settings.project,
    ]
    cmd.extend(args)
    return cmd


def run_compose(
    settings: IntegrationSettings,
    *args: str,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    _require_docker()
    return subprocess.run(
        compose_cmd(settings, *args),
        cwd=str(COMPOSE_FILE.parent),
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def up_infra(settings: IntegrationSettings) -> None:
    run_compose(
        settings,
        "up",
        "-d",
        "--build",
        "postgres",
        "redis",
        "etcd",
        "minio",
        "milvus",
    )


def up_apis(settings: IntegrationSettings) -> None:
    run_compose(settings, "up", "-d", "--build", "api-a", "api-b")


def up_workers(settings: IntegrationSettings, *names: str) -> None:
    targets = names or ("index-worker-a", "index-worker-b")
    # Recreate ensures killed containers come back with current env/mounts.
    run_compose(settings, "up", "-d", "--build", "--force-recreate", *targets)


def stop_service(settings: IntegrationSettings, name: str, *, kill: bool = False) -> None:
    if kill:
        # Force-stop without graceful app cleanup.
        run_compose(settings, "kill", name, check=False)
    run_compose(settings, "stop", name, check=False)


def start_service(settings: IntegrationSettings, name: str) -> None:
    run_compose(settings, "start", name, check=False)
    # Fresh container if previously removed
    run_compose(settings, "up", "-d", "--no-deps", name, check=False)


def down_all(settings: IntegrationSettings, *, volumes: bool = True) -> None:
    args = ["down", "--remove-orphans"]
    if volumes:
        args.append("-v")
    run_compose(settings, *args, check=False)


def service_inspect(settings: IntegrationSettings, name: str) -> dict[str, Any]:
    result = run_compose(settings, "ps", "--format", "json", name, check=False)
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    if not lines:
        return {}
    # compose v2 may emit one JSON object per line or a JSON array
    try:
        parsed = json.loads(lines[0] if len(lines) == 1 else f"[{','.join(lines)}]")
    except json.JSONDecodeError:
        return {"raw": result.stdout}
    if isinstance(parsed, list):
        return parsed[0] if parsed else {}
    return parsed


def container_id(settings: IntegrationSettings, name: str) -> str:
    info = service_inspect(settings, name)
    return str(info.get("ID") or info.get("Container") or info.get("Name") or "")


def logs(settings: IntegrationSettings, name: str, *, tail: int = 200) -> str:
    result = run_compose(settings, "logs", "--no-color", "--tail", str(tail), name, check=False)
    return (result.stdout or "") + (result.stderr or "")


def wait_http_ok(url: str, *, timeout_seconds: float = 180.0) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body.strip().startswith("{") else {"raw": body}
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
