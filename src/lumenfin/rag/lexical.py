"""Shared lexical tokenization for hybrid keyword score + rerank.

Chinese-friendly: CJK runs keep the full phrase plus character n-grams, and
financial ZH↔EN synonym groups expand both sides before overlap scoring.
No model / cross-encoder dependency — safe for CI and offline demos.
"""

from __future__ import annotations

import re

_LATIN = re.compile(r"[a-zA-Z0-9_]+")
_CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")

# Normalize common multi-word / punctuated phrases before tokenization.
_PHRASE_NORMS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"operating\s+margins?", re.IGNORECASE), " operating_margin "),
    (re.compile(r"supply\s+chains?", re.IGNORECASE), " supply_chain "),
    (
        re.compile(r"r\s*&\s*d|research\s+and\s+development|\brnd\b", re.IGNORECASE),
        " r_and_d ",
    ),
    (re.compile(r"r_and_d_intensity|rd\s+intensity|研发\s*强度", re.IGNORECASE), " r_and_d 强度 "),
)

# Bidirectional synonym groups (any hit expands to the whole group).
_SYNONYM_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"operating_margin", "营业利润率", "经营利润率", "营业利润"}),
    frozenset({"r_and_d", "rd", "研发", "研发强度", "研发费用", "研发投入"}),
    frozenset({"revenue", "sales", "收入", "营收", "营业收入", "销售额"}),
    frozenset({"ebitda", "息税折旧摊销前利润"}),
    frozenset({"margin", "利润率", "毛利率", "净利率"}),
    frozenset({"profitability", "盈利", "盈利能力", "获利"}),
    frozenset({"supply_chain", "供应链", "供应链风险", "代工"}),
    frozenset({"risk", "风险", "风险点"}),
    frozenset({"compare", "comparison", "对比", "比较", "对照"}),
    frozenset({"intensity", "强度"}),
    frozenset({"apple", "苹果"}),
    frozenset({"microsoft", "微软"}),
    frozenset({"nvidia", "英伟达"}),
    frozenset({"tsmc", "台积电"}),
    frozenset({"oracle", "甲骨文"}),
)

# Quick lookup: token -> groups that contain it (or that it substring-matches).
_SYNONYM_BY_TOKEN: dict[str, frozenset[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _member in _group:
        _SYNONYM_BY_TOKEN[_member] = _group


def _normalize_phrases(text: str) -> str:
    normalized = text or ""
    for pattern, replacement in _PHRASE_NORMS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def tokenize_text(text: str) -> set[str]:
    """Latin words + CJK full runs + CJK bi/trigrams."""
    lowered = _normalize_phrases(text).lower()
    tokens: set[str] = set()
    for match in _LATIN.finditer(lowered):
        token = match.group(0)
        if len(token) > 1 or token.isdigit():
            tokens.add(token)
    for match in _CJK_RUN.finditer(lowered):
        run = match.group(0)
        tokens.add(run)
        if len(run) == 1:
            continue
        for i in range(len(run) - 1):
            tokens.add(run[i : i + 2])
        if len(run) >= 3:
            for i in range(len(run) - 2):
                tokens.add(run[i : i + 3])
    return tokens

def expand_synonyms(tokens: set[str]) -> set[str]:
    """Close tokens under financial ZH↔EN synonym groups."""
    if not tokens:
        return set()
    expanded = set(tokens)
    for token in tokens:
        group = _SYNONYM_BY_TOKEN.get(token)
        if group:
            expanded |= group
            continue
        # Substring match for multi-char CJK / compound tokens.
        if len(token) < 2:
            continue
        for member, group in _SYNONYM_BY_TOKEN.items():
            if len(member) < 2:
                continue
            if member in token or token in member:
                expanded |= group
    return expanded


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of expanded query tokens found in expanded document tokens."""
    query_tokens = expand_synonyms(tokenize_text(query))
    text_tokens = expand_synonyms(tokenize_text(text))
    if not query_tokens or not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / len(query_tokens)


def query_has_any(query_tokens: set[str], *needles: str) -> bool:
    """True if any needle appears in tokens or as a synonym-expanded hit."""
    expanded = expand_synonyms(query_tokens)
    return any(needle.lower() in expanded for needle in needles)
