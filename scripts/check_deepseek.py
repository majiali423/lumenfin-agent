"""Quick DeepSeek connectivity diagnostic (does not print full API key).

Uses the same dotenv / AppConfig load path as formal runtime:
process env wins; conflicting process vs project .env values fail fast.
Never calls ``load_dotenv(override=True)``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.env_bootstrap import (  # noqa: E402
    EnvConflictError,
    announce_credential_sources,
    assert_no_env_conflicts,
    bootstrap_dotenv,
)
from lumenfin.llm import LLMSettings  # noqa: E402


def diag_key(key: str) -> str:
    """Coarse key hygiene only — never echo credential contents."""
    if not key:
        return "EMPTY"
    if key.strip() != key:
        return "HAS_WHITESPACE"
    if key.startswith(('"', "'")) or key.endswith(('"', "'")):
        return "HAS_QUOTES"
    if "your-key" in key.lower():
        return "STILL PLACEHOLDER"
    if not key.startswith("sk-"):
        return "BAD_PREFIX"
    if any(ord(c) > 127 or c in "\r\n\t" for c in key):
        return "HAS_WEIRD_CHARS"
    return f"OK len={len(key)}"


def main() -> int:
    print("project root:", ROOT)
    print(".env exists:", (ROOT / ".env").exists())
    print("config load path: lumenfin.env_bootstrap.bootstrap_dotenv + LLMSettings.from_env")

    try:
        bootstrap_dotenv(root=ROOT, announce=False, strict_conflicts=True)
        assert_no_env_conflicts(root=ROOT)
        announce_credential_sources(root=ROOT)
        settings = LLMSettings.from_env()
    except EnvConflictError as exc:
        print(f"\nFAIL: {exc}")
        return 1

    key = (settings.api_key or "").strip()
    base = settings.base_url
    model = settings.model
    key_diag = diag_key(key)
    print("DEEPSEEK_API_KEY:", key_diag)
    print("DEEPSEEK_BASE_URL:", base)
    print("DEEPSEEK_MODEL:", model)
    print("provider fingerprint:", f"deepseek/{model} @ {base}")

    if not key_diag.startswith("OK"):
        print(f"\nFAIL: fix DEEPSEEK_API_KEY (source layer above) first ({key_diag})")
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
        if resp.status_code in {401, 403}:
            print("HINT: 401/403 means DEEPSEEK_API_KEY is wrong — not MAS_API_KEY.")
            print("HINT: If process env shadows .env, unset DEEPSEEK_API_KEY in the shell.")
        return 1
    except Exception as exc:
        print("FAIL exception:", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
