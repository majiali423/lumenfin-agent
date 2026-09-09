from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REQUIRED_ONLINE_SETTINGS = (
    "POSTGRES_PASSWORD",
    "MINIO_ROOT_USER",
    "MINIO_ROOT_PASSWORD",
    "DASHSCOPE_API_KEY",
    "DEEPSEEK_API_KEY",
    "SEC_USER_AGENT",
)
AUTH_SETTINGS = ("MAS_API_KEY", "MAS_API_KEY_PRINCIPALS")
DEFAULT_READY_URL = "http://127.0.0.1:8000/ready"
DEFAULT_APP_URL = "http://127.0.0.1:8000/"


def _load_online_environment() -> dict[str, str]:
    """Match Compose's .env + process environment precedence without logging values."""
    from dotenv import dotenv_values

    values = {
        key: str(value)
        for key, value in dotenv_values(ROOT / ".env").items()
        if value is not None
    }
    values.update({key: value for key, value in os.environ.items() if value})
    return values


def _missing_online_settings(values: Mapping[str, str]) -> list[str]:
    missing = [key for key in REQUIRED_ONLINE_SETTINGS if not values.get(key, "").strip()]
    if not any(values.get(key, "").strip() for key in AUTH_SETTINGS):
        missing.append("MAS_API_KEY (or MAS_API_KEY_PRINCIPALS)")
    return missing


def _validate_auth_settings(values: Mapping[str, str]) -> str | None:
    principals = values.get("MAS_API_KEY_PRINCIPALS", "").strip()
    if not principals:
        return None
    try:
        parsed = json.loads(principals)
    except json.JSONDecodeError:
        return "MAS_API_KEY_PRINCIPALS must be valid JSON"
    if not isinstance(parsed, dict) or not parsed:
        return "MAS_API_KEY_PRINCIPALS must be a non-empty JSON object"
    return None


def _compose_prefix() -> list[str] | None:
    docker = shutil.which("docker")
    if docker:
        result = subprocess.run(
            [docker, "compose", "version"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return [docker, "compose"]

    legacy = shutil.which("docker-compose")
    if legacy:
        return [legacy]

    if os.name == "nt":
        bundled = Path(r"C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe")
        if bundled.is_file():
            return [str(bundled)]
    return None


def _compose_command(prefix: Sequence[str], *, observability: bool) -> list[str]:
    command = [*prefix, "-f", str(ROOT / "docker-compose.yml")]
    if observability:
        command.extend(["-f", str(ROOT / "docker-compose.observability.yml")])
    return command


def _docker_is_ready() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    result = subprocess.run(
        [docker, "info"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def _start_docker_desktop(timeout_seconds: int = 120) -> bool:
    if os.name != "nt":
        return False
    executable = Path(r"C:\Program Files\Docker\Docker\Docker Desktop.exe")
    if not executable.is_file():
        return False
    print("Docker Desktop is not ready; starting it now ...")
    subprocess.Popen(
        [str(executable)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if _docker_is_ready():
            return True
        time.sleep(2)
    return False


def _wait_for_http(url: str, timeout_seconds: int) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(2)
    return False


def _print_urls(*, observability: bool) -> None:
    print("\nLumenFin online stack is ready:")
    print(f"  Web/API:    {DEFAULT_APP_URL}")
    if observability:
        print("  Grafana:    http://127.0.0.1:3000/")
        print("  Prometheus: http://127.0.0.1:9090/")
    print("Enter MAS_API_KEY from .env in the web page when prompted.")


def start_online(args: argparse.Namespace) -> int:
    values = _load_online_environment()
    missing = _missing_online_settings(values)
    auth_error = _validate_auth_settings(values)
    if missing or auth_error:
        print("Online configuration is incomplete. Update .env before starting.", file=sys.stderr)
        if missing:
            print("Missing: " + ", ".join(missing), file=sys.stderr)
        if auth_error:
            print(auth_error, file=sys.stderr)
        return 2

    prefix = _compose_prefix()
    if prefix is None:
        print(
            "Docker Compose was not found. Install/start Docker Desktop and retry.",
            file=sys.stderr,
        )
        return 2

    compose = _compose_command(prefix, observability=not args.without_observability)
    config_check = subprocess.run(
        [*compose, "config", "--quiet"], cwd=ROOT, check=False
    )
    if config_check.returncode != 0:
        print("Compose configuration validation failed.", file=sys.stderr)
        return config_check.returncode

    if not _docker_is_ready():
        if args.check_only:
            print(
                "Online configuration and Compose checks passed, but Docker engine is not ready.",
                file=sys.stderr,
            )
            return 2
        if args.no_docker_desktop_start or not _start_docker_desktop():
            print("Docker engine is not ready. Start Docker Desktop and retry.", file=sys.stderr)
            return 2

    if args.check_only:
        print("Online configuration, Docker, and Compose checks passed.")
        return 0

    up = [*compose, "up", "-d"]
    if not args.no_build:
        up.append("--build")
    print("Starting PostgreSQL, Redis, Milvus, API, workers, and web UI ...")
    if not args.without_observability:
        print("Prometheus and Grafana are included.")
    result = subprocess.run(up, cwd=ROOT, check=False)
    if result.returncode != 0:
        print("The online stack did not start successfully.", file=sys.stderr)
        return result.returncode

    print("Waiting for API readiness ...")
    if not _wait_for_http(DEFAULT_READY_URL, args.timeout):
        print("API readiness timed out. Current service status:", file=sys.stderr)
        subprocess.run([*compose, "ps"], cwd=ROOT, check=False)
        print("Inspect logs with: python start_api.py logs", file=sys.stderr)
        return 1

    _print_urls(observability=not args.without_observability)
    if not args.no_browser:
        webbrowser.open(DEFAULT_APP_URL)
    return 0


def show_status(args: argparse.Namespace) -> int:
    prefix = _compose_prefix()
    if prefix is None:
        print("Docker Compose was not found.", file=sys.stderr)
        return 2
    compose = _compose_command(prefix, observability=not args.without_observability)
    return subprocess.run([*compose, "ps"], cwd=ROOT, check=False).returncode


def show_logs(args: argparse.Namespace) -> int:
    prefix = _compose_prefix()
    if prefix is None:
        print("Docker Compose was not found.", file=sys.stderr)
        return 2
    compose = _compose_command(prefix, observability=not args.without_observability)
    services = ["lumenfin-api", "lumenfin-worker", "lumenfin-index-worker"]
    return subprocess.run(
        [*compose, "logs", "--tail", str(args.log_lines), *services],
        cwd=ROOT,
        check=False,
    ).returncode


def serve_api() -> int:
    """Run only the API process; this is the Docker container entrypoint."""
    import uvicorn

    from lumenfin.api.app import create_app
    from lumenfin.config import AppConfig
    from lumenfin.rag.profiles import apply_showcase_rag_env
    from lumenfin.stdio import configure_stdio_utf8

    configure_stdio_utf8()
    if os.getenv("APP_ENV", "dev").strip().lower() != "test":
        apply_showcase_rag_env(overwrite=False)
    config = AppConfig.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the complete LumenFin online demo stack."
    )
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("online", "serve", "status", "logs"),
        default="online",
        help="online is the default; serve runs only the API process",
    )
    parser.add_argument("--no-build", action="store_true", help="reuse existing images")
    parser.add_argument("--no-browser", action="store_true", help="do not open the web UI")
    parser.add_argument(
        "--without-observability",
        action="store_true",
        help="skip Prometheus and Grafana",
    )
    parser.add_argument(
        "--no-docker-desktop-start",
        action="store_true",
        help="do not try to start Docker Desktop automatically on Windows",
    )
    parser.add_argument("--check-only", action="store_true", help="validate without starting")
    parser.add_argument("--timeout", type=int, default=300, help="readiness timeout in seconds")
    parser.add_argument("--log-lines", type=int, default=120, help="lines shown by logs mode")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "serve":
        return serve_api()
    if args.mode == "status":
        return show_status(args)
    if args.mode == "logs":
        return show_logs(args)
    return start_online(args)


if __name__ == "__main__":
    raise SystemExit(main())
