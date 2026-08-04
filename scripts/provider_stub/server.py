"""Deterministic local fault provider for Phase 3.3A.

Scenarios are selected via ``X-LumenFin-Scenario`` header (test-only).
Production clients must not send this header by default.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "requests": [],
    "scenario_counters": {},
}


def reset_state() -> None:
    with _LOCK:
        _STATE["requests"] = []
        _STATE["scenario_counters"] = {}


def snapshot() -> dict[str, Any]:
    with _LOCK:
        return {
            "request_count": len(_STATE["requests"]),
            "requests": list(_STATE["requests"]),
            "scenario_counters": dict(_STATE["scenario_counters"]),
        }


class FaultHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            return json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {"_raw": raw.decode("utf-8", errors="replace")}

    def _scenario(self) -> str:
        header = (self.headers.get("X-LumenFin-Scenario") or "").strip()
        if header:
            return header
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        values = query.get("scenario") or []
        return (values[0] if values else "success").strip() or "success"

    def _record(self, *, scenario: str, status: int, path: str) -> int:
        with _LOCK:
            counter = int(_STATE["scenario_counters"].get(scenario) or 0) + 1
            _STATE["scenario_counters"][scenario] = counter
            _STATE["requests"].append(
                {
                    "ts": time.time(),
                    "path": path,
                    "scenario": scenario,
                    "status": status,
                    "n": counter,
                }
            )
            return counter

    def _send(self, status: int, body: Any, *, headers: dict[str, str] | None = None) -> None:
        payload = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(200, {"status": "ok"})
            return
        if parsed.path == "/__stub__/stats":
            self._send(200, snapshot())
            return
        if parsed.path == "/__stub__/reset":
            reset_state()
            self._send(200, {"reset": True})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        scenario = self._scenario()
        body = self._read_json()
        if parsed.path.endswith("/chat/completions"):
            self._handle_chat(scenario, body)
            return
        if parsed.path.endswith("/embeddings"):
            self._handle_embeddings(scenario, body)
            return
        self._record(scenario=scenario, status=404, path=parsed.path)
        self._send(404, {"error": "not found"})

    def _handle_chat(self, scenario: str, body: dict[str, Any]) -> None:
        n = self._record(scenario=scenario, status=0, path="/chat/completions")
        if scenario == "always_503":
            self._send(503, {"error": "unavailable"})
            return
        if scenario == "503_then_success":
            if n < 3:
                self._send(503, {"error": "unavailable"})
                return
            self._send(200, _chat_ok("recovered-after-503"))
            return
        if scenario == "429_then_success":
            if n < 2:
                self._send(429, {"error": "rate limited"}, headers={"Retry-After": "1"})
                return
            self._send(200, _chat_ok("recovered-after-429"))
            return
        if scenario == "permanent_400":
            self._send(400, {"error": "bad request"})
            return
        if scenario == "timeout":
            time.sleep(5.0)
            self._send(200, _chat_ok("too-late"))
            return
        if scenario == "slow_success":
            time.sleep(1.5)
            self._send(200, _chat_ok("slow-ok"))
            return
        if scenario == "malformed_json":
            payload = b"{not-json"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if scenario == "empty_completion":
            self._send(
                200,
                {
                    "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 0},
                },
            )
            return
        if scenario == "connection_close":
            self.close_connection = True
            return
        self._send(200, _chat_ok("ok"))

    def _handle_embeddings(self, scenario: str, body: dict[str, Any]) -> None:
        texts = body.get("input") or []
        if isinstance(texts, str):
            texts = [texts]
        dim = int(body.get("dimensions") or 8)
        n = self._record(scenario=scenario, status=0, path="/embeddings")
        if scenario == "embedding_count_mismatch":
            self._send(
                200,
                {
                    "data": [{"index": 0, "embedding": [0.1] * dim}],
                },
            )
            return
        if scenario == "embedding_dimension_mismatch":
            self._send(
                200,
                {
                    "data": [
                        {"index": i, "embedding": [0.1] * (dim + 1)} for i, _ in enumerate(texts)
                    ]
                },
            )
            return
        if scenario == "always_503":
            self._send(503, {"error": "unavailable"})
            return
        if scenario == "503_then_success" and n < 3:
            self._send(503, {"error": "unavailable"})
            return
        self._send(
            200,
            {
                "data": [{"index": i, "embedding": [0.01 * (i + 1)] * dim} for i, _ in enumerate(texts)]
            },
        )


def _chat_ok(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def serve(host: str = "0.0.0.0", port: int = 18090) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), FaultHandler)
    return server


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Phase 3.3A deterministic provider stub")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18090)
    args = parser.parse_args()
    reset_state()
    server = serve(args.host, args.port)
    print(f"provider-stub listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
