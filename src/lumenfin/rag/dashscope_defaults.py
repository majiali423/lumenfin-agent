"""DashScope embedding defaults shared by production RAG and FinanceBench eval.

This module must stay import-side-effect free: no HTTP clients, no dotenv, no
provider construction.
"""

from __future__ import annotations

import os

DEFAULT_DASHSCOPE_EMBEDDING_MODEL = "text-embedding-v4"


def resolved_dashscope_embedding_model(explicit: str | None = None) -> str:
    """Return the effective DashScope embedding model.

    Explicit constructor/CLI values win, then ``DASHSCOPE_EMBEDDING_MODEL``,
    then :data:`DEFAULT_DASHSCOPE_EMBEDDING_MODEL`. An explicit empty string
    is treated as unset so the default still applies.
    """
    return (
        explicit
        or os.getenv("DASHSCOPE_EMBEDDING_MODEL")
        or DEFAULT_DASHSCOPE_EMBEDDING_MODEL
    ).strip()
