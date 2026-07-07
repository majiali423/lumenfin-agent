"""Quick DeepSeek connectivity diagnostic (does not print full API key)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv, find_dotenv

ROOT = Path(__file__).resolve().parents[1]
env_path = ROOT / ".env"


def diag_key(key: str) -> str:
    if not key:
        return "EMPTY"
    if key.strip() != key:
        return "HAS_WHITESPACE (leading/trailing spaces)"
    if key.startswith(('"', "'")) or key.endswith(('"', "'")):
        return "HAS_QUOTES around value"
    if "your-key" in key.lower():
        return "STILL PLACEHOLDER"
    if not key.startswith("sk-"):
        return "BAD_PREFIX (should start with sk-)"
    weird = [hex(ord(c)) for c in key if ord(c) > 127 or c in "\r\n\t"]
    if weird:
        return f"HAS_WEIRD_CHARS {weird[:5]}"
    return f"OK len={len(key)} prefix={key[:7]}... suffix=...{key[-4:]}"


def main() -> int:
    print("project root:", ROOT)
    print(".env exists:", env_path.exists())
    print("find_dotenv:", find_dotenv())

    load_dotenv(env_path, override=True)
    key = os.getenv("DEEPSEEK_API_KEY", "")
    base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    print("DEEPSEEK_API_KEY:", diag_key(key))
    print("DEEPSEEK_BASE_URL:", base)
    print("DEEPSEEK_MODEL:", model)

    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPSEEK_API_KEY"):
                val = line.split("=", 1)[1] if "=" in line else ""
                print("raw .env value:", diag_key(val))

    if not key or "your-key" in key.lower():
        print("\nFAIL: fix DEEPSEEK_API_KEY in .env first")
        return 1

    url = f"{base.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with exactly: OK"},
        ],
        "max_tokens": 5,
    }
    print("\nPOST", url)
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(url, headers=headers, json=payload)
        print("HTTP status:", resp.status_code)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print("SUCCESS, reply:", content[:80])
            return 0
        print("FAIL body:", resp.text[:300])
        return 1
    except Exception as exc:
        print("FAIL exception:", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
