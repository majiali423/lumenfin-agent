"""Verify DashScope embedding connectivity (does not print the full API key)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
load_dotenv(ROOT / ".env", override=True)


def _diag_key(key: str) -> str:
    if not key:
        return "EMPTY"
    if key.strip() != key:
        return "HAS_WHITESPACE"
    if "your-key" in key.lower() or key.endswith("..."):
        return "PLACEHOLDER"
    if not key.startswith("sk-"):
        return "BAD_PREFIX"
    return f"OK len={len(key)} prefix={key[:4]}..."


def main() -> int:
    from lumenfin.rag.embeddings import DashScopeEmbeddingProvider, build_embedding_provider

    provider_name = (os.getenv("MAS_EMBEDDING_PROVIDER") or "").strip().lower()
    key = (os.getenv("DASHSCOPE_API_KEY") or "").strip()
    model = (os.getenv("DASHSCOPE_EMBEDDING_MODEL") or "text-embedding-v3").strip()
    dim = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSION") or os.getenv("MAS_EMBEDDING_DIMENSION") or "1024")
    base = (os.getenv("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()

    print("project root:", ROOT)
    print("MAS_EMBEDDING_PROVIDER:", provider_name or "<unset>")
    print("DASHSCOPE_API_KEY:", _diag_key(key))
    print("model:", model)
    print("dimension:", dim)
    print("base_url:", base)

    if provider_name not in {"dashscope", "aliyun", "alibaba"}:
        print("WARN: provider is not dashscope — still testing DashScope client directly.")
    if not _diag_key(key).startswith("OK"):
        print("FAIL: fix DASHSCOPE_API_KEY first")
        return 1

    try:
        provider = DashScopeEmbeddingProvider(api_key=key, model=model, dimension=dim, base_url=base)
        built = build_embedding_provider(provider_name or "dashscope", dimension=dim)
        print("build_embedding_provider:", type(built).__name__, "dim=", built.dimension)

        vectors = provider.embed(
            [
                "Apple FY2025 revenue and EBITDA margin analysis.",
                "苹果供应链风险与研发投入。",
            ]
        )
        print("SUCCESS: got", len(vectors), "vectors; lens=", [len(v) for v in vectors])
        print("sample L2-ish magnitude[0]=", round(sum(x * x for x in vectors[0]) ** 0.5, 4))
        return 0
    except Exception as exc:
        print("FAIL:", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
