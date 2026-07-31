from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .documents import normalize_metric_hints_to_billion_usd
from .market_data import DEFAULT_TICKER_MAP
from .metrics_schema import get_fundamental, normalize_market_data, set_fundamental
from .fundamentals import is_plausible_revenue_billion_usd
from .reporting import annotate_upload_period_meta


class SafeExpressionEvaluator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Load,
        ast.Name,
        ast.Constant,
        ast.UnaryOp,
        ast.USub,
    )

    def __init__(self, variables: dict[str, float]) -> None:
        self.variables = variables

    def visit(self, node: ast.AST) -> float:
        if not isinstance(node, self.allowed_nodes):
            raise ValueError(f"Unsafe node detected: {type(node).__name__}")
        return super().visit(node)

    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("Unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("Unsupported unary operator")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.variables:
            raise KeyError(node.id)
        return self.variables[node.id]

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise ValueError("Only numeric constants are allowed")
        return float(node.value)


def safe_execute_formula(formula: str, variables: dict[str, float]) -> float:
    tree = ast.parse(formula, mode="eval")
    evaluator = SafeExpressionEvaluator(variables)
    return round(evaluator.visit(tree), 4)


def resolve_safe_formula(formula: str, variables: dict[str, float], backend: str = "local") -> float:
    if backend == "mcp":
        from .mcp_bridge import compute_ratio_via_mcp

        return compute_ratio_via_mcp(formula, variables)
    return safe_execute_formula(formula, variables)


KNOWN_ALIASES = {
    "tesla": "Tesla",
    "特斯拉": "Tesla",
    "amazon": "Amazon",
    "亚马逊": "Amazon",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "谷歌": "Alphabet",
    "meta": "Meta",
    "meta platforms": "Meta",
    "facebook": "Meta",
    "nvidia": "NVIDIA",
    "英伟达": "NVIDIA",
    "amd": "AMD",
    "tencent": "Tencent",
    "腾讯": "Tencent",
    "腾讯控股": "Tencent",
    "apple": "Apple",
    "苹果": "Apple",
    "microsoft": "Microsoft",
    "微软": "Microsoft",
    "tsla": "Tesla",
    "nvda": "NVIDIA",
    "tsmc": "TSMC",
    "台积电": "TSMC",
    "taiwan semiconductor": "TSMC",
    "samsung": "Samsung",
    "三星": "Samsung",
    "byd": "BYD",
    "比亚迪": "BYD",
    "broadcom": "Broadcom",
    "avgo": "Broadcom",
    "alibaba": "Alibaba",
    "阿里巴巴": "Alibaba",
    "oracle": "Oracle",
    "甲骨文": "Oracle",
    "shopify": "Shopify",
    "block": "Block",
    "softbank": "SoftBank",
    "softbank group": "SoftBank",
    "软银": "SoftBank",
}


_AST_INPUT_KEYS = ("revenue", "ebitda", "operating_income", "r_and_d")


def _merge_document_market_data_with_live(
    document_md: dict[str, Any] | None,
    live_md: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    """Financial Grounding merge: keep document values; fill missing AST keys from live.

    Issuer-scoped only — callers must not pass peer companyfacts. Document wins on
    overlap so upload-extracted figures are not overwritten by SEC/Yahoo.
    """
    from .metrics_schema import get_fundamental, normalize_market_data, set_fundamental

    merged = normalize_market_data(dict(document_md or {}))
    live_n = normalize_market_data(dict(live_md or {}))
    filled: list[str] = []
    for key in _AST_INPUT_KEYS:
        if get_fundamental(merged, key) is not None:
            continue
        live_val = get_fundamental(live_n, key)
        if live_val is None:
            continue
        set_fundamental(merged, key, live_val)
        filled.append(key)
    return merged, filled


def retrieve_company_payload(
    company: str,
    include_appendix: bool = False,
    document_contexts: list[dict[str, Any]] | None = None,
    *,
    allow_sample_data: bool = True,
    ticker: str | None = None,
    fetch_live_fundamentals: bool = False,
    fetch_sec_fundamentals: bool = False,
    prefer_uploaded_only: bool = False,
    prefer_fiscal_year: int | None = None,
) -> dict[str, Any]:
    """Build a company payload from documents / SEC / Yahoo / sample.

    Preference order (Financial Grounding Layer):
    1) document-extracted metrics when AST-computable (revenue + ebitda/OI/R&D)
    2) SEC EDGAR companyfacts gap-fill for the **issuer** only (when enabled)
    3) Yahoo annual income statement gap-fill (when enabled)
    4) sample_db (when allow_sample_data=True)

    Partial upload hints no longer short-circuit SEC/Yahoo: ``bool(market_data)``
    alone is insufficient — only ``has_computable_fundamentals`` wins early.

    When ``prefer_uploaded_only`` is True, steps 2–4 are skipped so sparse uploads
    fail loud instead of silently backfilling from live providers.
    """
    doc_contexts = document_contexts or []
    document_payload = _payload_from_documents(company, doc_contexts, include_appendix=include_appendix)
    # Upload "present for this company" only when the issuer is explicitly detected
    # (empty detected_companies must not attribute the PDF to every peer).
    upload_present = any(
        company in (doc.get("detected_companies") or []) for doc in doc_contexts
    ) or bool(document_payload.get("market_data")) or bool(
        document_payload.get("earnings_call_quotes")
    )
    # Financial Grounding: complete document AST set wins; partial hints do not block SEC.
    if has_computable_fundamentals(document_payload):
        result = dict(document_payload)
        meta = dict(result.get("fundamentals_meta") or {})
        meta.update(
            {
                "upload_present": upload_present,
                "upload_had_computable_metrics": True,
                "live_fallback_used": False,
                "prefer_uploaded_only": prefer_uploaded_only,
                "grounding_layer": "document_ast_complete",
            }
        )
        meta = annotate_upload_period_meta(
            meta,
            document_contexts=doc_contexts,
            company=company,
            prefer_fiscal_year=prefer_fiscal_year,
        )
        result["fundamentals_meta"] = meta
        return _finalize_company_payload(result)

    # Upload-only mode: never invent numbers from SEC/Yahoo/sample.
    if prefer_uploaded_only:
        result = dict(document_payload)
        # Narrative excerpts may still exist; without computable metrics this is not a structured source.
        result["structured_source"] = "none"
        meta = dict(result.get("fundamentals_meta") or {})
        meta.update(
            {
                "upload_present": upload_present,
                "upload_had_computable_metrics": False,
                "live_fallback_used": False,
                "prefer_uploaded_only": True,
                "grounding_layer": "prefer_uploaded_only_refused",
                "fallback_reason": (
                    "prefer_uploaded_only=true; refused SEC/Yahoo/sample backfill because "
                    "uploaded materials lacked AST-computable revenue/EBITDA/R&D."
                ),
            }
        )
        result["fundamentals_meta"] = meta
        return _finalize_company_payload(result)

    from .market_data import DEFAULT_TICKER_MAP

    symbol = (ticker or DEFAULT_TICKER_MAP.get(company) or company).strip()
    provider_errors: list[dict[str, Any]] = []

    def _annotate_fallback(
        merged: dict[str, Any],
        *,
        provider: str,
        filled_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        meta = dict(merged.get("fundamentals_meta") or {})
        if upload_present:
            meta.update(
                {
                    "upload_present": True,
                    "upload_had_computable_metrics": False,
                    "live_fallback_used": True,
                    "prefer_uploaded_only": False,
                    "grounding_layer": "issuer_sec_gap_fill",
                    "sec_filled_keys": list(filled_keys or []),
                    "fallback_reason": (
                        "Uploaded materials lacked AST-computable revenue/EBITDA/R&D; "
                        f"issuer structured fundamentals filled from {provider}"
                        + (
                            f" (filled: {', '.join(filled_keys)})."
                            if filled_keys
                            else "."
                        )
                    ),
                }
            )
            merged["fundamentals_meta"] = meta
        return merged

    def _merge_live(live: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        """Attach upload narrative/supply-chain and gap-fill AST market_data."""
        result = dict(live)
        filled: list[str] = []
        doc_md = document_payload.get("market_data") or {}
        live_md = live.get("market_data") or {}
        if doc_md:
            merged_md, filled = _merge_document_market_data_with_live(doc_md, live_md)
            result["market_data"] = merged_md
        provenance = dict(document_payload.get("fundamental_provenance") or {})
        live_source = str(live.get("structured_source") or "sec_companyfacts")
        live_period = None
        live_meta = live.get("fundamentals_meta") or {}
        if isinstance(live_meta, dict) and live_meta.get("fiscal_year") is not None:
            live_period = f"FY{live_meta.get('fiscal_year')}"
        for key in _AST_INPUT_KEYS:
            if key in provenance:
                continue
            if get_fundamental(result.get("market_data") or {}, key) is None:
                continue
            if key in filled or not doc_md or get_fundamental(doc_md, key) is None:
                provenance[key] = {
                    "source": live_source,
                    "confidence": "high",
                    "period": live_period,
                }
        for key in _AST_INPUT_KEYS:
            if key in provenance:
                continue
            if get_fundamental(doc_md, key) is not None:
                provenance[key] = {
                    "source": "document_extracted",
                    "confidence": "high",
                    "period": live_period,
                }
        if provenance:
            result["fundamental_provenance"] = provenance
        if document_payload.get("document_observations"):
            result["document_observations"] = dict(document_payload["document_observations"])
        if document_payload.get("earnings_call_quotes"):
            result["earnings_call_quotes"] = list(document_payload["earnings_call_quotes"])
        if document_payload.get("supply_chain", {}).get("signals"):
            result["supply_chain"] = dict(document_payload["supply_chain"])
        if document_payload.get("source_documents"):
            result["source_documents"] = list(document_payload.get("source_documents") or [])
        return result, filled

    if fetch_sec_fundamentals:
        from .sec_fundamentals import fetch_sec_companyfacts_fundamentals

        sec_live = fetch_sec_companyfacts_fundamentals(
            symbol,
            errors=provider_errors,
            prefer_fiscal_year=prefer_fiscal_year,
        )
        if sec_live and sec_live.get("market_data"):
            merged, filled = _merge_live(sec_live)
            merged = _annotate_fallback(merged, provider="sec_companyfacts", filled_keys=filled)
            if prefer_fiscal_year is not None:
                meta = dict(merged.get("fundamentals_meta") or {})
                meta.setdefault("requested_fiscal_year", prefer_fiscal_year)
                if meta.get("period_alignment") is None:
                    used = meta.get("fiscal_year")
                    meta["period_alignment"] = (
                        "exact"
                        if used is not None and int(used) == int(prefer_fiscal_year)
                        else "fallback_latest"
                    )
                merged["fundamentals_meta"] = meta
            if provider_errors:
                merged["provider_errors"] = list(provider_errors)
            return _finalize_company_payload(merged)

    if fetch_live_fundamentals:
        from .fundamentals import fetch_yahoo_fundamentals

        yahoo_live = fetch_yahoo_fundamentals(symbol, errors=provider_errors)
        if yahoo_live and yahoo_live.get("market_data"):
            merged, filled = _merge_live(yahoo_live)
            merged = _annotate_fallback(
                merged, provider="yahoo_fundamentals", filled_keys=filled
            )
            if provider_errors:
                merged["provider_errors"] = list(provider_errors)
            return _finalize_company_payload(merged)

    has_sample_data = allow_sample_data and company in SAMPLE_FINANCIAL_DATA
    if has_sample_data:
        payload = SAMPLE_FINANCIAL_DATA[company]
        result = {
            "market_data": dict(payload["market_data"]),
            "supply_chain": dict(payload["supply_chain"]),
            "earnings_call_quotes": list(payload["earnings_call_quotes"]),
            "structured_source": "sample_db",
            "fundamentals_meta": {"provider": "sample_db", "period": "demo_latest"},
        }
        filled: list[str] = []
        doc_md = document_payload.get("market_data") or {}
        if doc_md:
            merged_md, filled = _merge_document_market_data_with_live(
                doc_md, result["market_data"]
            )
            result["market_data"] = merged_md
        if document_payload.get("earnings_call_quotes"):
            result["earnings_call_quotes"] = list(document_payload["earnings_call_quotes"])
        if document_payload.get("supply_chain", {}).get("signals"):
            result["supply_chain"] = dict(document_payload["supply_chain"])
        if include_appendix:
            result["appendix"] = dict(payload["appendix"])
        if provider_errors:
            result["provider_errors"] = list(provider_errors)
        result = _annotate_fallback(result, provider="sample_db", filled_keys=filled)
        return _finalize_company_payload(result)

    if provider_errors:
        document_payload = dict(document_payload)
        document_payload["provider_errors"] = list(provider_errors)
    return _finalize_company_payload(document_payload)


def _finalize_company_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize fundamental keys on every retrieve path."""
    result = dict(payload)
    result["market_data"] = normalize_market_data(result.get("market_data"))
    appendix = result.get("appendix")
    if isinstance(appendix, dict) and appendix:
        result["appendix"] = normalize_market_data(appendix)
    return result


def _payload_from_documents(
    company: str,
    doc_contexts: list[dict[str, Any]],
    *,
    include_appendix: bool,
) -> dict[str, Any]:
    from .documents import (
        extract_metric_amounts_for_company,
        extract_metric_hint_meta,
        is_trusted_ast_amount,
        normalize_metric_hints_to_billion_usd,
    )

    market_data: dict[str, float] = {}
    supply_chain_signals: list[str] = []
    earnings_quotes_list: list[str] = []
    document_observations: dict[str, Any] = {"metric_hints": {}, "metric_hint_meta": {}}
    fundamental_provenance: dict[str, dict[str, Any]] = {}

    for doc in doc_contexts:
        if not _document_applies_to_company(doc, company):
            continue
        detected = doc.get("detected_companies", [])
        text = doc.get("text", "")
        excerpt = doc.get("excerpt", "")[:3000]

        scoped = (doc.get("per_company_metric_hints") or {}).get(company) or {}
        scoped_meta = (doc.get("per_company_metric_hint_meta") or {}).get(company) or {}
        if not scoped_meta and text:
            amounts = extract_metric_amounts_for_company(text, company)
            from .documents import amount_to_meta

            scoped_meta = {key: amount_to_meta(amount) for key, amount in amounts.items()} if amounts else {}
            if amounts and not scoped:
                from .documents import _compatibility_hints

                scoped = _compatibility_hints(amounts)
        if not scoped and text:
            from .documents import extract_metric_hints_for_company

            scoped = extract_metric_hints_for_company(text, company)
        hint_source = scoped or (
            doc.get("metric_hints", {}) if len(detected) <= 1 else {}
        )
        hint_meta = scoped_meta or (
            doc.get("metric_hint_meta", {}) if len(detected) <= 1 else {}
        )
        # Keep low-confidence projections observable without promoting into AST.
        projected = normalize_metric_hints_to_billion_usd(
            dict(hint_source), text=text, hint_meta=hint_meta or None
        )
        document_observations["metric_hints"].update(projected)
        document_observations["metric_hint_meta"].update(hint_meta or {})

        for key, value in projected.items():
            if key not in {"revenue", "ebitda", "r_and_d", "operating_income"} and not (
                key.endswith("_2025")
                and key.replace("_2025", "") in {"revenue", "ebitda", "r_and_d", "operating_income"}
            ):
                continue
            base = key.replace("_2025", "") if key.endswith("_2025") else key
            meta = (hint_meta or {}).get(key) or (hint_meta or {}).get(base)
            provided_hints = doc.get("metric_hints") or {}
            caller_supplied = key in provided_hints or base in provided_hints
            explicit_meta = (doc.get("metric_hint_meta") or {}).get(key) or (
                doc.get("metric_hint_meta") or {}
            ).get(base)
            if not explicit_meta:
                per_co = (doc.get("per_company_metric_hint_meta") or {}).get(company) or {}
                explicit_meta = per_co.get(key) or per_co.get(base)
            if (
                caller_supplied
                and not explicit_meta
                and not is_trusted_ast_amount(meta)
            ):
                # Explicit caller floats without extraction metadata are already on the
                # shared billion scale. Do not let opportunistic unitless text parses
                # (e.g. "10-K", bare "400") block promotion of those provided hints.
                meta = {
                    "raw_value": float(value),
                    "raw_scale": None,
                    "currency": "USD",
                    "normalized_value": float(value),
                    "normalized_unit": "billion_usd",
                    "normalization_source": "provider_metadata",
                    "confidence": "high",
                    "period_hint": None,
                    "is_normalized": True,
                }
                document_observations["metric_hint_meta"][key] = meta
            elif not meta and text:
                extracted = extract_metric_hint_meta(text, metric=base)
                if extracted:
                    meta = extracted
            if not is_trusted_ast_amount(meta):
                continue
            if base == "revenue" and not is_plausible_revenue_billion_usd(value):
                continue
            set_fundamental(market_data, key, float(meta.get("normalized_value", value)))
            fundamental_provenance[base] = {
                "source": "document_extracted",
                "confidence": meta.get("confidence"),
                "normalization_source": meta.get("normalization_source"),
                "period": meta.get("period_hint"),
            }

        if excerpt:
            earnings_quotes_list.append(excerpt[:500])

        lowered_text = text.lower()
        if has_supply_chain_signal(lowered_text):
            supply_chain_signals.append("PDF 文档中包含供应链相关讨论。")

    return {
        "market_data": market_data,
        "supply_chain": {
            "risk_level": "medium" if supply_chain_signals else "unknown",
            "signals": supply_chain_signals or (["PDF 文档中未检测到明确供应链信号。"] if doc_contexts else []),
        },
        "earnings_call_quotes": earnings_quotes_list,
        "structured_source": "document_extracted" if market_data else "none",
        "document_observations": document_observations,
        "fundamental_provenance": fundamental_provenance,
        "metric_hint_meta": document_observations.get("metric_hint_meta") or {},
        **({"appendix": {}} if include_appendix else {}),
    }


_SUPPLY_CHAIN_PHRASES = (
    "supply chain risk",
    "供应链风险",
    "supply constraint",
    "supply constraints",
    "logistics risk",
    "supplier concentration",
    "供应商集中",
    "packaging capacity",
    "foundry",
)


def has_supply_chain_signal(text: str) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in _SUPPLY_CHAIN_PHRASES)


# Backward-compatible private alias.
_has_supply_chain_signal = has_supply_chain_signal


def _document_applies_to_company(doc: dict[str, Any], company: str) -> bool:
    """Same issuer scoping as retrieval payload assembly: require explicit detection."""
    detected = doc.get("detected_companies") or []
    return company in detected


def _append_unique_company(companies: list[str], name: str) -> None:
    if name and name not in companies:
        companies.append(name)


def is_plausible_company_label(name: str) -> bool:
    """Drop OCR/LLM mojibake labels that break live provenance gates."""
    raw = (name or "").strip()
    if len(raw) < 2 or len(raw) > 80:
        return False
    if "\ufffd" in raw or "?" in raw:
        return False
    if raw in KNOWN_ALIASES or raw in KNOWN_ALIASES.values():
        return True
    lowered = raw.lower()
    if lowered in KNOWN_ALIASES:
        return True
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9 .,&'-]{1,60}", raw):
        return True
    if re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9]{2,20}", raw):
        return True
    return False


def _canonicalize_company_name(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        return raw
    lowered = raw.lower()
    if raw in KNOWN_ALIASES:
        return KNOWN_ALIASES[raw]
    if lowered in KNOWN_ALIASES:
        return KNOWN_ALIASES[lowered]
    return raw


def canonicalize_companies(companies: list[str]) -> list[str]:
    """Dedupe company labels via aliases (e.g. 腾讯控股 -> Tencent)."""
    canonical: list[str] = []
    for company in companies:
        cleaned = _canonicalize_company_name(str(company))
        if not is_plausible_company_label(cleaned):
            continue
        _append_unique_company(canonical, cleaned)
    return canonical


def has_computable_fundamentals(payload: dict[str, Any] | None) -> bool:
    """True when AST quant can compute at least one core ratio from structured inputs."""
    market = (payload or {}).get("market_data") or {}
    revenue = get_fundamental(market, "revenue")
    if revenue in (None, 0):
        return False
    return any(
        get_fundamental(market, key) is not None
        for key in ("ebitda", "operating_income", "r_and_d")
    )


AST_RATIO_KEYS = ("ebitda_margin", "r_and_d_intensity", "operating_margin")


def classify_quant_status(metrics: dict[str, float] | None) -> str:
    """Classify per-company quant output for peer-comparison coverage."""
    values = metrics or {}
    if any(key in values for key in AST_RATIO_KEYS):
        return "ast_ok"
    if values:
        return "market_only"
    return "uncomputable"


def build_coverage_matrix(
    companies: list[str],
    retrieved_docs: dict[str, dict[str, Any]],
    financial_metrics: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, Any]]:
    matrix: dict[str, dict[str, Any]] = {}
    for company in companies:
        payload = retrieved_docs.get(company) or {}
        metrics = (financial_metrics or {}).get(company) or {}
        has_structured = has_computable_fundamentals(payload)
        if metrics:
            quant_status = classify_quant_status(metrics)
            comparable = quant_status == "ast_ok"
        elif has_structured:
            quant_status = "pending"
            comparable = True
        elif payload.get("market_data") or (payload.get("live_market") or {}).get("current_price"):
            quant_status = "pending_market"
            comparable = False
        else:
            quant_status = "uncomputable"
            comparable = False
        matrix[company] = {
            "structured_source": str(payload.get("structured_source") or "none"),
            "has_computable_fundamentals": has_structured,
            "quant_status": quant_status,
            "ast_ratios": quant_status == "ast_ok",
            "comparable": comparable,
        }
    return matrix


def is_partial_compare_gap(companies: list[str], coverage_matrix: dict[str, dict[str, Any]]) -> bool:
    """True when a multi-company run has both comparable and non-comparable peers."""
    if len(companies) <= 1:
        return False
    comparable = [company for company in companies if (coverage_matrix.get(company) or {}).get("comparable")]
    return bool(comparable) and len(comparable) < len(companies)


def non_comparable_companies(companies: list[str], coverage_matrix: dict[str, dict[str, Any]]) -> list[str]:
    return [company for company in companies if not (coverage_matrix.get(company) or {}).get("comparable")]

def _extract_companies_via_llm(query: str, llm_client: Any) -> list[str]:
    try:
        prompt = llm_client.chat(
            system_prompt="你是一个公司名称提取器。从用户查询中提取所有被提及的公司名称。"
            '返回 JSON 格式: {"companies": ["公司1", "公司2"]}。只返回 JSON，不要其他内容。',
            user_prompt=query,
            temperature=0.0,
            max_tokens=100,
        )
        prompt_clean = prompt.strip()
        if prompt_clean.startswith("```"):
            prompt_clean = prompt_clean.split("\n", 1)[-1].rsplit("\n", 1)[0]
        data = json.loads(prompt_clean)
        return [str(name) for name in data.get("companies", []) if name]
    except Exception:
        return []


def _alias_mentioned(alias: str, lowered: str, original: str = "") -> bool:
    """Match aliases without short English substring false positives (block ⊂ blockchain)."""
    token = (alias or "").strip()
    if not token:
        return False
    if re.search(r"[\u4e00-\u9fff]", token):
        return token in original or token in lowered
    if " " in token or len(token) >= 6:
        return token in lowered or token in original.lower()
    return bool(re.search(rf"(?<![a-z0-9]){re.escape(token)}(?![a-z0-9])", lowered))


alias_mentioned = _alias_mentioned


def english_token_in_text(token: str, text: str) -> bool:
    """Boundary-aware English token match; CJK / multi-word phrases use containment."""
    return _alias_mentioned(token, text.lower(), text)


def any_token_in_text(tokens: tuple[str, ...] | list[str], text: str) -> bool:
    return any(english_token_in_text(token, text) for token in tokens)


def extract_companies_from_query(
    query: str,
    document_contexts: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
) -> list[str]:
    """Extract company names from query using sample data, PDF context, and LLM."""
    companies: list[str] = []
    lowered = query.lower()

    # 1. Check sample data for direct mentions (name detection only; sample values gated elsewhere)
    for company in SAMPLE_FINANCIAL_DATA:
        if _alias_mentioned(company.lower(), lowered, query):
            _append_unique_company(companies, company)

    # 2. Check known aliases + shared COMPANY_HINTS (CJK aliases match original text too)
    from .documents import COMPANY_HINTS

    for alias, name in {**KNOWN_ALIASES, **COMPANY_HINTS}.items():
        if _alias_mentioned(alias, lowered, query):
            _append_unique_company(companies, name)

    # 3. Collect companies detected in uploaded PDFs
    doc_contexts = document_contexts or []
    for doc in doc_contexts:
        for company in doc.get("detected_companies", []):
            _append_unique_company(companies, _canonicalize_company_name(str(company)))

    # 4. Merge LLM extraction so comparative queries do not stop at the first sample hit
    if llm_client:
        for company in _extract_companies_via_llm(query, llm_client):
            _append_unique_company(companies, _canonicalize_company_name(str(company)))

    # Canonicalize + dedupe (e.g. LLM emits 腾讯控股 while alias already added Tencent)
    return canonicalize_companies(companies) if companies else (
        [_canonicalize_company_name(Path(doc.get("filename", "")).stem)
         for doc in doc_contexts
         if doc.get("filename") and len(Path(doc.get("filename", "")).stem) < 50][:1]
        or []
    )

def _ticker_to_companies() -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    for company, ticker in DEFAULT_TICKER_MAP.items():
        reverse.setdefault(str(ticker).upper(), []).append(company)
    return reverse


def derive_target_symbols(companies: list[str], query: str) -> dict[str, str]:
    """Map companies → tickers (curated map, then SEC directory, then explicit tokens)."""
    from .ticker_resolve import resolve_ticker_for_company

    symbols: dict[str, str] = {}
    for company in companies:
        resolved = resolve_ticker_for_company(company, query=query, allow_network=True)
        if resolved:
            # Always key by the pipeline company label so retrieval lookups match.
            symbols[company] = resolved.ticker
        else:
            symbols[company] = DEFAULT_TICKER_MAP.get(company, company)

    explicit_tokens = re.findall(r"\b(?:ticker|symbol)\s*[:=]\s*([A-Z]{1,5})\b", query, flags=re.IGNORECASE)
    explicit_tokens.extend(re.findall(r"\(([A-Z]{1,5})\)", query))
    if not explicit_tokens:
        return symbols

    ticker_owners = _ticker_to_companies()
    company_set = set(companies)
    for token in explicit_tokens:
        upper = token.upper()
        owners = [name for name in ticker_owners.get(upper, []) if name in company_set]
        if len(owners) == 1:
            symbols[owners[0]] = upper
            continue
        matched = [name for name, sym in symbols.items() if str(sym).upper() == upper and name in company_set]
        if len(matched) == 1:
            continue
        if len(companies) == 1:
            symbols[companies[0]] = upper
            continue
    return symbols


def summarize_document_context(document_contexts: list[dict[str, Any]], company: str) -> dict[str, Any]:
    related_docs = []
    metric_hints: dict[str, float] = {}
    joined_text_parts: list[str] = []
    for doc in document_contexts:
        if not _document_applies_to_company(doc, company):
            continue
        detected = doc.get("detected_companies") or []
        related_docs.append(
            {
                "document_id": doc.get("document_id"),
                "filename": doc.get("filename"),
                "excerpt": doc.get("excerpt", "")[:1200],
            }
        )
        joined_text_parts.append(str(doc.get("text") or doc.get("excerpt") or ""))
        scoped = (doc.get("per_company_metric_hints") or {}).get(company) or {}
        hint_source = scoped or (doc.get("metric_hints", {}) if len(detected) <= 1 else {})
        for metric_name, value in hint_source.items():
            metric_hints.setdefault(metric_name, value)
    metric_hints = normalize_metric_hints_to_billion_usd(
        metric_hints,
        text="\n".join(joined_text_parts),
    )
    return {"source_documents": related_docs, "metric_hints": metric_hints}


def analyze_sentiment(quotes: list[str]) -> dict[str, Any]:
    positive_markers = [
        "optimistic", "confident", "healthy", "durable", "constructive", "resilience",
        "strong", "growth", "accelerating", "efficiency", "gain", "record", "robust",
        "momentum", "remain confident", "remain optimistic", "positive",
    ]
    caution_markers = [
        "risk", "constraints", "pressure", "remain", "challenge", "volatility",
        "uncertain", "headwind", "concern", "exposure", "supply chain",
        "concentration", "regulatory", "despite",
    ]
    joined = " ".join(quotes).lower()
    positive_hits = sum(1 for marker in positive_markers if marker in joined)
    caution_hits = sum(1 for marker in caution_markers if marker in joined)
    if positive_hits == 0 and caution_hits == 0:
        label = "neutral"
    elif positive_hits >= caution_hits:
        label = "bullish"
    else:
        label = "cautious"
    return {
        "label": label,
        "positive_hits": positive_hits,
        "caution_hits": caution_hits,
    }


_WEAK_QUOTE_MARKERS = (
    "profile generation pending",
    "profile generation skipped",
    "not available",
    "n/a",
    "no quote",
    "placeholder",
    "文档已上传",
    "请基于 pdf",
    "请基于pdf",
)


def quotes_are_weak_for_llm(quotes: list[str] | None, *, min_chars: int = 80) -> bool:
    """True when quotes are empty, tiny, or placeholder — skip deep LLM tone analysis."""
    cleaned = [str(q).strip() for q in (quotes or []) if str(q).strip()]
    if not cleaned:
        return True
    joined = " ".join(cleaned)
    if len(joined) < min_chars:
        return True
    lower = joined.lower()
    if any(marker in lower for marker in _WEAK_QUOTE_MARKERS):
        return True
    if len(cleaned) == 1 and len(cleaned[0]) < 40:
        return True
    return False


def validate_report(report: str) -> list[str]:
    """Lightweight bilingual template check (EN or ZH markers)."""
    text = report or ""
    findings: list[str] = []
    has_disclaimer = (
        "风险免责声明" in text
        or re.search(r"(?i)disclaimer", text) is not None
        or re.search(r"(?i)not investment advice", text) is not None
    )
    has_provenance = (
        "数据来源" in text
        or re.search(r"(?i)data sources?", text) is not None
        or re.search(r"(?i)methodology", text) is not None
        or "structured_source" in text
    )
    if not has_disclaimer:
        findings.append("缺少风险免责声明。")
    if not has_provenance:
        findings.append("缺少数据来源标注。")
    return findings


def analyze_sentiment_deep(quotes: list[str], llm_client: Any | None = None) -> dict[str, Any]:
    """Deep sentiment analysis using LLM when quotes are substantive."""
    basic = analyze_sentiment(quotes)

    if quotes_are_weak_for_llm(quotes):
        return {
            "label": basic["label"],
            "positive_hits": basic["positive_hits"],
            "caution_hits": basic["caution_hits"],
            "confidence_score": 2,
            "key_themes": [],
            "risk_flags": [],
            "strategic_priority": "",
            "llm_skipped": True,
            "skip_reason": "weak_quotes",
        }

    if llm_client and quotes:
        try:
            joined_quotes = "\n".join(quotes[:5])[:2000]
            response = llm_client.chat(
                system_prompt=(
                    "You are a management tone analysis expert. Analyze the provided earnings-call quotes and "
                    "return JSON format: {\"overall_tone\": \"bullish/cautious/neutral\", "
                    "\"confidence_score\": 0-10, \"key_themes\": [\"theme1\",\"theme2\"], "
                    "\"risk_flags\": [\"risk1\"], \"strategic_priority\": \"priority\"}"
                ),
                user_prompt=f"Earnings call quotes:\n{joined_quotes}",
                temperature=0.1,
                max_tokens=250,
            )
            clean = response.strip()
            if clean.startswith("```"):
                clean = clean.split("\n", 1)[-1].rsplit("\n", 1)[0]
            deep = json.loads(clean)
            return {
                "label": deep.get("overall_tone", basic["label"]),
                "positive_hits": basic["positive_hits"],
                "caution_hits": basic["caution_hits"],
                "confidence_score": deep.get("confidence_score", 5),
                "key_themes": deep.get("key_themes", []),
                "risk_flags": deep.get("risk_flags", []),
                "strategic_priority": deep.get("strategic_priority", ""),
                "llm_skipped": False,
            }
        except Exception:
            pass
    return {
        **basic,
        "llm_skipped": True,
        "skip_reason": "llm_unavailable_or_failed",
    }


def calculate_derived_ratios(market_data: dict[str, float]) -> dict[str, float]:
    """Calculate additional financial ratios from available market data."""
    ratios: dict[str, float] = {}
    revenue = get_fundamental(market_data, "revenue")
    ebitda = get_fundamental(market_data, "ebitda")
    r_and_d = get_fundamental(market_data, "r_and_d")
    op_income = get_fundamental(market_data, "operating_income")

    if revenue and revenue > 0:
        if ebitda:
            ratios["ebitda_margin"] = round(ebitda / revenue, 4)
        if r_and_d:
            ratios["r_and_d_intensity"] = round(r_and_d / revenue, 4)
        if op_income:
            ratios["operating_margin"] = round(op_income / revenue, 4)
        # Estimated ratios based on industry averages
        if ebitda:
            ratios["estimated_net_margin"] = round((ebitda * 0.55) / revenue, 4)
            ratios["estimated_fcf_margin"] = round((ebitda * 0.40) / revenue, 4)
    return ratios


def build_chart_data(
    companies: list[str],
    financial_metrics: dict[str, dict[str, float]],
    sentiment_analysis: dict[str, dict[str, Any]],
    risk_scores: dict[str, dict[str, float]],
    audit_log: list[dict[str, str]],
) -> dict[str, Any]:
    """Build structured chart data for frontend visualization."""
    colors = ["#2563eb", "#7c3aed", "#059669", "#d97706", "#dc2626", "#0891b2"]

    # 1. Financial metrics comparison bar chart
    metric_keys = ["ebitda_margin", "r_and_d_intensity", "operating_margin", "estimated_net_margin"]
    metric_labels = {"ebitda_margin": "EBITDA Margin %", "r_and_d_intensity": "R&D Intensity %",
                     "operating_margin": "Operating Margin %", "estimated_net_margin": "Est. Net Margin %"}
    metrics_comparison = {"labels": companies, "datasets": []}
    for idx, key in enumerate(metric_keys):
        data_points = []
        for c in companies:
            val = financial_metrics.get(c, {}).get(key)
            data_points.append(round(val * 100, 2) if val is not None else None)
        if any(v is not None for v in data_points):
            metrics_comparison["datasets"].append({
                "label": metric_labels.get(key, key),
                "data": data_points,
                "backgroundColor": colors[idx % len(colors)] + "BB",
                "borderColor": colors[idx % len(colors)],
                "borderWidth": 1,
            })

    # 2. Risk radar chart
    risk_dimensions = ["financial_risk", "operational_risk", "market_risk", "regulatory_risk", "supply_chain_risk"]
    risk_labels = {"financial_risk": "Financial", "operational_risk": "Operational", "market_risk": "Market",
                   "regulatory_risk": "Regulatory", "supply_chain_risk": "Supply Chain"}
    risk_radar = {"labels": [risk_labels.get(d, d) for d in risk_dimensions], "datasets": []}
    for idx, company in enumerate(companies):
        scores = risk_scores.get(company, {})
        data = [scores.get(d) for d in risk_dimensions]
        risk_radar["datasets"].append({
            "label": company,
            "data": [d if d is not None else 5 for d in data],
            "backgroundColor": colors[idx % len(colors)] + "28",
            "borderColor": colors[idx % len(colors)],
            "borderWidth": 2,
            "pointBackgroundColor": colors[idx % len(colors)],
        })

    # 3. Sentiment distribution doughnut
    sentiment_data = {"labels": [], "datasets": [{"data": [], "backgroundColor": []}]}
    tone_counts: dict[str, int] = {}
    tone_colors = {"bullish": "#059669", "cautious": "#d97706", "neutral": "#64748b", "unknown": "#94a3b8"}
    for company, sentiment in sentiment_analysis.items():
        label = sentiment.get("label", "unknown")
        tone_counts[label] = tone_counts.get(label, 0) + 1
    for tone, count in tone_counts.items():
        sentiment_data["labels"].append(tone.capitalize())
        sentiment_data["datasets"][0]["data"].append(count)
        sentiment_data["datasets"][0]["backgroundColor"].append(tone_colors.get(tone, "#94a3b8"))

    # 4. Agent workflow timeline
    agent_timeline = [{"step": e.get("step", ""), "status": e.get("status", ""), "detail": e.get("detail", "")}
                      for e in audit_log]

    return {
        "metrics_comparison": metrics_comparison,
        "risk_radar": risk_radar,
        "sentiment_distribution": sentiment_data,
        "agent_timeline": agent_timeline,
        "colors": colors,
    }


def generate_scenario_analysis(metrics: dict[str, float], company: str) -> dict[str, Any]:
    """Generate scenario analysis (base/bull/bear) with probabilities."""
    ebitda_margin = metrics.get("ebitda_margin", 0.15)
    rd_intensity = metrics.get("r_and_d_intensity", 0.05)

    base_growth = 0.08 if ebitda_margin > 0.25 else 0.05
    bull_growth = base_growth * 1.8
    bear_growth = base_growth * 0.3

    base_prob = 0.50
    bull_prob = 0.30 if rd_intensity > 0.06 else 0.20
    bear_prob = 1.0 - base_prob - bull_prob

    return {
        "base_case": {"revenue_growth": f"{base_growth:.0%}", "probability": f"{base_prob:.0%}",
                      "narrative": f"Under moderate macro growth, {company} sustains current market share and margins."},
        "bull_case": {"revenue_growth": f"{bull_growth:.0%}", "probability": f"{bull_prob:.0%}",
                      "narrative": f"Technology upside or policy tailwinds drive above-consensus growth and share gains for {company}."},
        "bear_case": {"revenue_growth": f"{bear_growth:.0%}", "probability": f"{bear_prob:.0%}",
                      "narrative": f"Macro slowdown or competitive pressure reduces revenue growth and compresses margins for {company}."},
    }


def parse_with_fallback(text: str) -> dict[str, Any]:
    """Parse JSON from LLM, handling markdown fences."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1].rsplit("\n", 1)[0]
    return json.loads(clean)
