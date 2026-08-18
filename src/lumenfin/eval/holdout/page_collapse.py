"""Eval-only page diversity helpers for ranked retrieval hits."""

from __future__ import annotations

from typing import Any


def normalize_document_identity(value: object) -> str:
    text = str(value or "").strip()
    if text.casefold().endswith(".pdf"):
        text = text[:-4]
    return text.casefold()


def page_key(hit: dict[str, Any]) -> tuple[str, int] | None:
    """Return a stable (document, 1-indexed page) identity when available."""
    doc_name = normalize_document_identity(
        hit.get("document_id") or hit.get("filename") or hit.get("doc_name")
    )
    page = hit.get("page")
    if not doc_name or page is None or page == "" or isinstance(page, bool):
        return None
    try:
        page_one = int(page)
    except (TypeError, ValueError):
        return None
    if page_one <= 0 or str(page_one) != str(page).strip():
        return None
    return (doc_name, page_one)


def _window(hits: list[dict[str, Any]], k: int | None) -> list[dict[str, Any]]:
    if k is None:
        return list(hits)
    if k < 0:
        raise ValueError("k must be >= 0")
    return list(hits[:k])


def collapse_to_unique_pages(
    hits: list[dict[str, Any]],
    *,
    k: int | None = None,
) -> list[dict[str, Any]]:
    """Keep the highest-ranked hit per known page and backfill to ``k``.

    Hits without a usable page identity remain in rank order because silently
    dropping an unidentified candidate would change recall.
    """
    if k is not None and k < 0:
        raise ValueError("k must be >= 0")
    if k == 0:
        return []
    collapsed: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for hit in hits:
        key = page_key(hit)
        if key is None:
            collapsed.append(hit)
        elif key in seen:
            continue
        else:
            seen.add(key)
            collapsed.append(hit)
        if k is not None and len(collapsed) >= k:
            break
    return collapsed


def unique_pages_top_k(hits: list[dict[str, Any]], *, k: int = 10) -> int:
    return len(
        {
            key
            for hit in _window(hits, k)
            if (key := page_key(hit)) is not None
        }
    )


def page_identity_coverage_top_k(
    hits: list[dict[str, Any]],
    *,
    k: int = 10,
) -> float:
    window = _window(hits, k)
    if not window:
        return 0.0
    known = sum(page_key(hit) is not None for hit in window)
    return round(known / len(window), 4)


def duplicate_page_occupancy(hits: list[dict[str, Any]], *, k: int = 10) -> float:
    window = _window(hits, k)
    if not window:
        return 0.0
    known_keys = [key for hit in window if (key := page_key(hit)) is not None]
    duplicates = len(known_keys) - len(set(known_keys))
    return round(duplicates / len(window), 4)
