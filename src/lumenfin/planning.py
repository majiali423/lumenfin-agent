"""Query planning: rule layer + one structured LLM extract + merge/validate.

Plan fields are the source of truth. Optional ``retrieval_query`` is a constrained
rewrite for RAG only and must mention the same companies; free-form query rewrite
is intentionally not used as the planner contract.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .llm import BaseLLMClient
from .skills import SKILL_REGISTRY
from .tools import (
    KNOWN_ALIASES,
    alias_mentioned,
    any_token_in_text,
    canonicalize_companies,
    extract_companies_from_query,
    parse_with_fallback,
)

# Prefer boundary-aware patterns over short substring tokens like "fy"/"year"
# (which false-positive on words such as "Clarify").
_CN_NUM = r"[一二三四五六七八九十两\d]+"
_TIME_RANGE_RE = re.compile(
    rf"(?:"
    rf"\b(?:19|20)\d{{2}}\s*[-–—~至到]\s*(?:19|20)\d{{2}}\b"
    rf"|\b(?:19|20)\d{{2}}\b"
    rf"|\bfy\s*(?:19|20)?\d{{2}}\b"
    rf"|\bq[1-4]\b(?:\s*(?:(?:19|20)\d{{2}}))?"
    rf"|\bh[12]\b(?:\s*(?:(?:19|20)\d{{2}}))?"
    rf"|\b(?:ttm|yoy|qoq|latest|annual)\b"
    rf"|\b(?:last|past|recent|these|those|prior)\s+{_CN_NUM}\s+years?\b"
    rf"|财年|年度|年报|季报|季度|去年|今年|前年"
    rf"|(?:最近|近|过去|这|前)\s*{_CN_NUM}\s*年"
    rf"|\b(?:this|last|prior)\s+year\b"
    rf"|\b(?:ytd|year[\s-]?to[\s-]?date)\b"
    rf")",
    re.IGNORECASE,
)

ALLOWED_INTENTS = (
    "document_financial_diligence",
    "comparative_financial_diligence",
    "risk_compliance_review",
    "financial_diligence",
)

ALLOWED_DIMENSIONS = (
    "profitability",
    "liquidity",
    "solvency",
    "r_and_d",
    "supply_chain",
    "sentiment",
    "compliance",
    "market",
    "document_evidence",
)

IntentName = Literal[
    "document_financial_diligence",
    "comparative_financial_diligence",
    "risk_compliance_review",
    "financial_diligence",
]


class TimeRangeStructure(BaseModel):
    raw: str = ""
    has_time: bool = False


class ConfidenceStructure(BaseModel):
    companies: float = 0.0
    time_range: float = 0.0
    prefer_uploaded_only: float = 0.0


class QueryStructureLLM(BaseModel):
    """Validated LLM extract for the planner (not the full QueryPlan contract)."""

    companies: list[str] = Field(default_factory=list)
    time_range: TimeRangeStructure = Field(default_factory=TimeRangeStructure)
    intent: IntentName = "financial_diligence"
    dimensions: list[str] = Field(default_factory=list)
    prefer_uploaded_only: bool = False
    confidence: ConfidenceStructure = Field(default_factory=ConfidenceStructure)
    retrieval_query: str = ""

    @field_validator("companies", mode="before")
    @classmethod
    def _clean_companies(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("companies must be a list")
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("dimensions", mode="before")
    @classmethod
    def _filter_dimensions(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("dimensions must be a list")
        cleaned: list[str] = []
        for item in value:
            name = str(item).strip()
            if name in ALLOWED_DIMENSIONS and name not in cleaned:
                cleaned.append(name)
        return cleaned

    @field_validator("retrieval_query", mode="before")
    @classmethod
    def _clean_retrieval_query(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()


_LLM_COMPANY_CONF_MIN = 0.55
_LLM_TIME_CONF_MIN = 0.60
_STRUCTURE_MAX_ATTEMPTS = 2


@dataclass(frozen=True)
class QueryPlan:
    normalized_query: str
    intent: str
    companies: list[str]
    analysis_dimensions: list[str]
    output_format: str
    required_skills: list[str]
    missing_fields: list[str]
    clarification_questions: list[str]
    planner_notes: list[str]
    time_range: str = ""
    retrieval_query: str = ""
    # When True: do not backfill structured fundamentals from SEC/Yahoo/sample.
    prefer_uploaded_only: bool = False
    query_companies: list[str] = field(default_factory=list)
    upload_companies: list[str] = field(default_factory=list)
    company_scope: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Require exclusivity — "using the uploaded table" alone is hybrid, not upload-only.
_UPLOAD_ONLY_EXCLUSIVE = (
    "uploaded note only",
    "uploaded file only",
    "uploaded materials only",
    "uploaded table only",
    "only the uploaded",
    "only uploaded",
    "using only the uploaded",
    "use only the uploaded",
    "using the uploaded note only",
    "using the uploaded file only",
    "using the uploaded materials only",
    "using the uploaded table only",
    "only from upload",
    "only use upload",
    "upload only",
    "uploads only",
    "do not use external",
    "don't use external",
    "no external data",
    "without external",
    "仅用上传",
    "只用上传",
    "仅基于上传",
    "只基于上传",
    "不要用外部",
    "不要外部数据",
    "勿用外部",
)


DIMENSION_KEYWORDS = {
    "profitability": ("profit", "margin", "ebitda", "盈利", "利润", "毛利", "净利"),
    "liquidity": ("liquidity", "cash", "现金", "流动性"),
    "solvency": ("debt", "leverage", "solvency", "负债", "偿债", "杠杆"),
    "r_and_d": ("r&d", "research", "研发", "创新"),
    "supply_chain": ("supply", "supplier", "供应链", "供应商"),
    "sentiment": ("sentiment", "tone", "management", "管理层", "语气", "措辞"),
    "compliance": ("compliance", "audit", "risk disclosure", "合规", "审计", "披露"),
    "market": ("market", "valuation", "price", "行情", "估值", "股价"),
}


def build_query_plan(
    query: str,
    document_contexts: list[dict[str, Any]] | None = None,
    llm_client: BaseLLMClient | None = None,
    user_clarification: dict[str, Any] | None = None,
) -> QueryPlan:
    document_contexts = document_contexts or []
    normalized_query = " ".join(query.split())

    # ── Rule layer (cheap, deterministic) ───────────────────────────
    with ThreadPoolExecutor(max_workers=4) as executor:
        intent_future = executor.submit(_detect_intent, normalized_query, document_contexts)
        company_future = executor.submit(_detect_query_companies_rules, normalized_query)
        dimension_future = executor.submit(_detect_dimensions, normalized_query, document_contexts)
        output_future = executor.submit(_detect_output_format, normalized_query)

        rule_intent = intent_future.result()
        rule_query_companies = company_future.result()
        rule_dimensions = dimension_future.result()
        output_format = output_future.result()

    rule_has_time = _has_time_range_signal(normalized_query)
    time_range_label = _rule_time_label(normalized_query) if rule_has_time else ""
    rule_prefer_uploaded_only = _detect_prefer_uploaded_only(normalized_query)
    upload_companies = _upload_companies(document_contexts)

    # ── One structured LLM extract when useful ──────────────────────
    structure: dict[str, Any] | None = None
    need_structure = bool(llm_client) and (
        (not rule_query_companies and not document_contexts)
        or not rule_has_time
        or rule_intent == "comparative_financial_diligence"
        or (bool(document_contexts) and not rule_prefer_uploaded_only)
    )
    if need_structure and llm_client is not None:
        structure = _extract_query_structure_via_llm(
            normalized_query,
            document_contexts=document_contexts,
            llm_client=llm_client,
        )

    query_companies, company_notes = _merge_query_companies(
        rule_query_companies,
        structure,
        query=normalized_query,
    )
    companies, company_scope, scope_notes, has_mismatch = _select_companies(
        query_companies=query_companies,
        upload_companies=upload_companies,
        user_clarification=user_clarification,
    )
    company_notes = company_notes + scope_notes
    intent = _merge_intent(rule_intent, structure)
    dimensions = _merge_dimensions(rule_dimensions, structure, document_contexts)
    has_time, time_range_label, time_notes = _merge_time_range(
        normalized_query,
        rule_has_time=rule_has_time,
        rule_label=time_range_label,
        structure=structure,
        document_contexts=document_contexts,
    )
    retrieval_query = _validated_retrieval_query(
        structure.get("retrieval_query") if structure else None,
        companies=companies,
        fallback_query=normalized_query,
        dimensions=dimensions,
    )
    prefer_uploaded_only, upload_mode_notes = _merge_prefer_uploaded_only(
        rule_prefer_uploaded_only,
        structure,
        has_documents=bool(document_contexts),
    )

    missing_fields = _detect_missing_fields(
        companies,
        document_contexts,
        has_time_signal=has_time,
        company_upload_mismatch=has_mismatch,
    )
    clarification_questions = _build_clarification_questions(
        missing_fields,
        query_companies=query_companies,
        upload_companies=upload_companies,
    )
    required_skills = _infer_required_skills(dimensions, has_documents=bool(document_contexts))
    planner_notes = _build_planner_notes(
        required_skills,
        missing_fields,
        document_contexts,
        company_notes=company_notes,
        time_notes=time_notes,
        used_llm_structure=bool(structure),
        retrieval_query=retrieval_query,
        prefer_uploaded_only=prefer_uploaded_only,
        upload_mode_notes=upload_mode_notes,
    )

    return QueryPlan(
        normalized_query=normalized_query,
        intent=intent,
        companies=companies,
        analysis_dimensions=dimensions,
        output_format=output_format,
        required_skills=required_skills,
        missing_fields=missing_fields,
        clarification_questions=clarification_questions,
        planner_notes=planner_notes,
        time_range=time_range_label,
        retrieval_query=retrieval_query,
        prefer_uploaded_only=prefer_uploaded_only,
        query_companies=query_companies,
        upload_companies=upload_companies,
        company_scope=company_scope,
    )


def _detect_intent(query: str, document_contexts: list[dict[str, Any]]) -> str:
    # User-requested compare/peer intent wins even when filings are attached.
    if any_token_in_text(("compare", "versus", "vs", "对比", "比较"), query):
        return "comparative_financial_diligence"
    if document_contexts or any_token_in_text(("pdf", "upload", "上传", "文件", "财报"), query):
        return "document_financial_diligence"
    if any_token_in_text(("risk", "compliance", "audit", "风险", "合规", "审计"), query):
        return "risk_compliance_review"
    return "financial_diligence"


def _detect_query_companies_rules(query: str) -> list[str]:
    """Companies named in the user query only (uploads handled separately)."""
    return canonicalize_companies(
        extract_companies_from_query(query, document_contexts=None, llm_client=None)
    )


def _detect_companies_rules(
    query: str,
    document_contexts: list[dict[str, Any]],
) -> list[str]:
    """Backward-compatible union of query + upload companies."""
    companies = _detect_query_companies_rules(query)
    for company in _upload_companies(document_contexts):
        if company not in companies:
            companies.append(company)
    return canonicalize_companies(companies)


def _upload_companies(document_contexts: list[dict[str, Any]]) -> list[str]:
    found: list[str] = []
    for doc in document_contexts or []:
        # Prefer issuer/primary entities over body peer mentions.
        candidates = doc.get("issuer_companies") or doc.get("detected_companies") or []
        for company in candidates:
            name = str(company).strip()
            if name and name not in found:
                found.append(name)
    return canonicalize_companies(found)


def _has_company_upload_mismatch(query_companies: list[str], upload_companies: list[str]) -> bool:
    """True when the user named issuers that are not present in the upload set."""
    query_set = set(query_companies)
    upload_set = set(upload_companies)
    return bool(query_set and upload_set and not query_set.issubset(upload_set))


def _resolve_companies_from_clarification(
    clarification: dict[str, Any] | None,
    *,
    query_companies: list[str],
    upload_companies: list[str],
) -> tuple[list[str] | None, str]:
    if not clarification:
        return None, ""
    raw = clarification.get("companies")
    if raw is None:
        raw = clarification.get("company")
    if raw is not None:
        if isinstance(raw, str):
            names = [raw]
        elif isinstance(raw, list):
            names = [str(x) for x in raw]
        else:
            names = [str(raw)]
        resolved = canonicalize_companies([n for n in names if str(n).strip()])
        if resolved:
            return resolved, "explicit"

    scope = str(
        clarification.get("company_scope")
        or clarification.get("scope")
        or clarification.get("resolve_mismatch")
        or ""
    ).lower().strip()
    if scope in {"uploaded", "upload", "use_uploaded", "documents", "doc", "file"}:
        return list(upload_companies), "uploaded"
    if scope in {"query", "use_query", "asked", "question"}:
        return list(query_companies), "query"
    if scope in {"both", "use_both", "all", "union"}:
        return canonicalize_companies(list(query_companies) + list(upload_companies)), "both"
    return None, ""


def _select_companies(
    *,
    query_companies: list[str],
    upload_companies: list[str],
    user_clarification: dict[str, Any] | None,
) -> tuple[list[str], str, list[str], bool]:
    notes: list[str] = []
    resolved, scope = _resolve_companies_from_clarification(
        user_clarification,
        query_companies=query_companies,
        upload_companies=upload_companies,
    )
    if resolved is not None:
        notes.append(f"Company scope resolved via clarification ({scope}): {resolved}.")
        return resolved, scope, notes, False

    mismatch = _has_company_upload_mismatch(query_companies, upload_companies)
    if mismatch:
        notes.append(
            "Company/upload mismatch: "
            f"query={query_companies} upload={upload_companies}; pausing for HITL."
        )
        # Keep query companies visible; do not silently analyze upload issuers.
        return list(query_companies), "mismatch", notes, True

    if query_companies:
        notes.append(f"Using query-named companies: {query_companies}.")
        return list(query_companies), "query", notes, False
    if upload_companies:
        notes.append(f"No query company; using upload-detected companies: {upload_companies}.")
        return list(upload_companies), "uploaded", notes, False
    return [], "", notes, False


def _has_explicit_company_signal(query: str, document_contexts: list[dict[str, Any]]) -> bool:
    lowered = query.lower()
    if any(alias_mentioned(company.lower(), lowered, query) for company in SAMPLE_FINANCIAL_DATA):
        return True
    from .documents import COMPANY_HINTS

    alias_keys = set(KNOWN_ALIASES) | set(COMPANY_HINTS)
    if any(alias_mentioned(alias, lowered, query) for alias in alias_keys):
        return True
    return any(doc.get("detected_companies") or doc.get("filename") for doc in document_contexts)


def _detect_dimensions(query: str, document_contexts: list[dict[str, Any]]) -> list[str]:
    dimensions = [
        dimension
        for dimension, keywords in DIMENSION_KEYWORDS.items()
        if any_token_in_text(keywords, query)
    ]
    if document_contexts and "document_evidence" not in dimensions:
        dimensions.append("document_evidence")
    if not dimensions:
        dimensions = ["profitability", "r_and_d", "supply_chain", "sentiment", "compliance"]
    if "compliance" not in dimensions:
        dimensions.append("compliance")
    return dimensions


def _detect_output_format(query: str) -> str:
    if any_token_in_text(("table", "表格"), query):
        return "table_summary"
    if any_token_in_text(("brief", "summary", "摘要", "简版"), query):
        return "executive_summary"
    return "research_report"


def _has_time_range_signal(query: str) -> bool:
    return bool(_TIME_RANGE_RE.search(query))


def _rule_time_label(query: str) -> str:
    match = _TIME_RANGE_RE.search(query)
    return match.group(0).strip() if match else ""


def _detect_prefer_uploaded_only(query: str) -> bool:
    lowered = query.lower()
    if any(phrase in lowered for phrase in _UPLOAD_ONLY_EXCLUSIVE):
        return True
    # "only" within a short window of upload wording
    if re.search(r"\bonly\b.{0,48}\bupload", lowered) or re.search(
        r"\bupload\w*.{0,48}\bonly\b", lowered
    ):
        return True
    if re.search(r"仅用|只用|仅基于|只基于", query) and re.search(
        r"上传|文件|材料|表格|pdf|PDF", query
    ):
        return True
    return False


def _merge_prefer_uploaded_only(
    rule_flag: bool,
    structure: dict[str, Any] | None,
    *,
    has_documents: bool,
) -> tuple[bool, list[str]]:
    notes: list[str] = []
    if rule_flag:
        notes.append("prefer_uploaded_only=true from exclusive query phrasing (no SEC/Yahoo/sample backfill).")
        return True, notes
    # Do not let the LLM alone force upload-only mode: phrases like "using the uploaded
    # table" are hybrid by default. Exclusivity must come from explicit wording.
    if structure and bool(structure.get("prefer_uploaded_only")):
        notes.append(
            "LLM suggested prefer_uploaded_only but was ignored without exclusive phrasing."
        )
    if has_documents:
        return False, notes
    return False, notes


def _extract_query_structure_via_llm(
    query: str,
    *,
    document_contexts: list[dict[str, Any]],
    llm_client: BaseLLMClient,
) -> dict[str, Any] | None:
    doc_hint = ""
    if document_contexts:
        names: list[str] = []
        for doc in document_contexts[:3]:
            names.extend(str(c) for c in (doc.get("detected_companies") or []))
            if doc.get("filename"):
                names.append(str(doc["filename"]))
        if names:
            doc_hint = f"Uploaded context hints: {', '.join(names[:12])}."

    system_prompt = (
        "You extract a structured finance analysis request. Return ONLY JSON with keys:\n"
        '{"companies":["..."],'
        '"time_range":{"raw":"...","has_time":true},'
        '"intent":"document_financial_diligence|comparative_financial_diligence|'
        'risk_compliance_review|financial_diligence",'
        '"dimensions":["profitability","liquidity","solvency","r_and_d","supply_chain",'
        '"sentiment","compliance","market"],'
        '"prefer_uploaded_only":false,'
        '"confidence":{"companies":0.0,"time_range":0.0,"prefer_uploaded_only":0.0},'
        '"retrieval_query":"short English retrieval query"}\n'
        "Rules:\n"
        "- companies: only entities clearly requested; use common English names when possible.\n"
        "- If the user did not name a company, companies must be [].\n"
        "- has_time=true only when a period is expressed (year, FY, last N years, 这两年, etc.).\n"
        "- dimensions: subset of the allowed list only.\n"
        "- prefer_uploaded_only=true ONLY when the user insists on uploaded materials alone "
        "(e.g. using the uploaded note / 仅用上传 / no external data). "
        "Uploading a file alone is NOT enough; default false so SEC/Yahoo may fill missing numbers.\n"
        "- retrieval_query must mention exactly the same companies (or be empty if companies=[]).\n"
        "- Do not invent filings, tickers, or companies absent from the user text/hints."
    )
    user_prompt = f"User query: {query}"
    if doc_hint:
        user_prompt += f"\n{doc_hint}"

    last_error = ""
    for attempt in range(_STRUCTURE_MAX_ATTEMPTS):
        attempt_prompt = user_prompt
        if last_error:
            attempt_prompt = (
                f"{user_prompt}\n"
                f"Previous output was invalid ({last_error}). "
                "Return ONLY valid JSON matching the schema; intent must be one of the allowed enums."
            )
        try:
            raw = llm_client.chat(
                system_prompt=system_prompt,
                user_prompt=attempt_prompt,
                temperature=0.0,
                max_tokens=280,
            )
            parsed = parse_with_fallback(raw)
            if not isinstance(parsed, dict):
                last_error = "parsed payload was not a JSON object"
                continue
            model = QueryStructureLLM.model_validate(parsed)
            return model.model_dump()
        except (ValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = str(exc).replace("\n", " ")[:220]
            continue
        except Exception as exc:  # noqa: BLE001 — network/LLM errors also trigger one retry
            last_error = f"{type(exc).__name__}: {exc}"[:220]
            continue
    return None


def _merge_query_companies(
    rule_companies: list[str],
    structure: dict[str, Any] | None,
    *,
    query: str = "",
) -> tuple[list[str], list[str]]:
    """Merge rule + LLM companies from the query text only (not uploads)."""
    notes: list[str] = []
    merged = list(rule_companies)
    if structure:
        conf = float(((structure.get("confidence") or {}).get("companies")) or 0.0)
        llm_companies = [str(c).strip() for c in (structure.get("companies") or []) if str(c).strip()]
        if llm_companies and conf >= _LLM_COMPANY_CONF_MIN:
            before = set(canonicalize_companies(merged))
            for name in llm_companies:
                if name in merged:
                    continue
                # Require the issuer to be grounded in the user query text, not only upload hints.
                if query and not _company_mentioned_in_query(name, query):
                    notes.append(f"LLM company ignored (not mentioned in query): {name}.")
                    continue
                merged.append(name)
            after = canonicalize_companies(merged)
            added = [c for c in after if c not in before]
            if added:
                notes.append(f"LLM structure added companies={added} (conf={conf:.2f}).")
            merged = after
        elif llm_companies and conf < _LLM_COMPANY_CONF_MIN:
            notes.append(f"LLM companies ignored due to low confidence ({conf:.2f}).")

    merged = canonicalize_companies(merged)
    # SEC ticker directory: resolve names, accept bare tickers in query, note unresolved.
    try:
        from .ticker_resolve import enrich_company_universe

        enriched, _symbols, resolve_notes = enrich_company_universe(
            merged,
            query=query,
            allow_network=True,
        )
        if resolve_notes:
            notes.extend(resolve_notes[:8])
        if enriched:
            merged = canonicalize_companies(enriched)
    except Exception:
        pass
    return merged, notes


def _company_mentioned_in_query(company: str, query: str) -> bool:
    lowered = query.lower()
    if alias_mentioned(company.lower(), lowered, query):
        return True
    from .documents import COMPANY_HINTS

    for alias, canonical in {**KNOWN_ALIASES, **COMPANY_HINTS}.items():
        if canonical == company and alias_mentioned(alias, lowered, query):
            return True
    # Bare ticker or parenthetical symbol grounding.
    upper = company.strip().upper()
    if re.fullmatch(r"[A-Z]{1,5}", upper) and re.search(rf"\b{re.escape(upper)}\b", query.upper()):
        return True
    if re.search(rf"\({re.escape(upper)}\)", query.upper()):
        return True
    # Multi-word SEC-style labels: require first token hit.
    first = company.strip().split()[0].lower() if company.strip() else ""
    if len(first) >= 4 and alias_mentioned(first, lowered, query):
        return True
    return False


def _merge_companies(
    rule_companies: list[str],
    document_contexts: list[dict[str, Any]],
    structure: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Legacy helper: query companies union upload companies."""
    query_companies, notes = _merge_query_companies(rule_companies, structure)
    upload = _upload_companies(document_contexts)
    return canonicalize_companies(query_companies + upload), notes


def _merge_intent(rule_intent: str, structure: dict[str, Any] | None) -> str:
    if not structure:
        return rule_intent
    candidate = str(structure.get("intent") or "").strip()
    if candidate not in ALLOWED_INTENTS:
        return rule_intent
    if rule_intent == "financial_diligence" and candidate != rule_intent:
        return candidate
    if rule_intent == "risk_compliance_review" and candidate == "comparative_financial_diligence":
        return candidate
    return rule_intent


def _merge_dimensions(
    rule_dimensions: list[str],
    structure: dict[str, Any] | None,
    document_contexts: list[dict[str, Any]],
) -> list[str]:
    dims = list(rule_dimensions)
    if structure:
        for item in structure.get("dimensions") or []:
            name = str(item).strip()
            if name in ALLOWED_DIMENSIONS and name not in dims:
                dims.append(name)
    if document_contexts and "document_evidence" not in dims:
        dims.append("document_evidence")
    if "compliance" not in dims:
        dims.append("compliance")
    if not dims:
        dims = ["profitability", "r_and_d", "supply_chain", "sentiment", "compliance"]
    return dims


def _merge_time_range(
    query: str,
    *,
    rule_has_time: bool,
    rule_label: str,
    structure: dict[str, Any] | None,
    document_contexts: list[dict[str, Any]],
) -> tuple[bool, str, list[str]]:
    notes: list[str] = []
    if document_contexts:
        return True, rule_label or "document_context", notes
    if rule_has_time:
        return True, rule_label or _rule_time_label(query), notes

    if structure:
        time_obj = structure.get("time_range") or {}
        conf = float(((structure.get("confidence") or {}).get("time_range")) or 0.0)
        has_time = bool(time_obj.get("has_time"))
        raw = str(time_obj.get("raw") or "").strip()
        if has_time and conf >= _LLM_TIME_CONF_MIN:
            notes.append(f"LLM structure accepted time_range={raw or 'unspecified'} (conf={conf:.2f}).")
            return True, raw, notes
        if has_time and conf < _LLM_TIME_CONF_MIN:
            notes.append(f"LLM time_range ignored due to low confidence ({conf:.2f}).")
    return False, "", notes


def _validated_retrieval_query(
    candidate: str | None,
    *,
    companies: list[str],
    fallback_query: str,
    dimensions: list[str],
) -> str:
    """Accept LLM retrieval_query only if it preserves company set; else build a safe focus string."""
    focus = f"{fallback_query} | focus: {', '.join(dimensions)}" if dimensions else fallback_query
    text = (candidate or "").strip()
    if not text:
        return focus
    if not companies:
        return focus
    lowered = text.lower()
    for company in companies:
        token = company.lower()
        if token not in lowered and not any(
            alias_mentioned(alias, lowered, text)
            for alias, name in KNOWN_ALIASES.items()
            if name == company
        ):
            return focus
    return text


def _detect_missing_fields(
    companies: list[str],
    document_contexts: list[dict[str, Any]],
    *,
    has_time_signal: bool,
    company_upload_mismatch: bool = False,
) -> list[str]:
    missing = []
    if company_upload_mismatch:
        missing.append("company_upload_mismatch")
    if not companies and not document_contexts:
        missing.append("company")
    if not document_contexts and not has_time_signal:
        missing.append("time_range")
    return missing


def _build_clarification_questions(
    missing_fields: list[str],
    *,
    query_companies: list[str] | None = None,
    upload_companies: list[str] | None = None,
) -> list[str]:
    questions = []
    if "company_upload_mismatch" in missing_fields:
        q = ", ".join(query_companies or []) or "(none)"
        u = ", ".join(upload_companies or []) or "(none)"
        questions.append(
            f"查询与上传公司不一致：查询 [{q}] / 上传 [{u}]。"
            "请选 company_scope=uploaded | query | both（或直接提供 companies）。"
        )
    if "company" in missing_fields:
        questions.append("请填写公司名称（例如 Apple、Microsoft）。")
    if "time_range" in missing_fields:
        questions.append("请填写时间范围或财年（例如 FY2025）。")
    return questions


def _infer_required_skills(dimensions: list[str], has_documents: bool) -> list[str]:
    skills = ["company_identification"]
    if has_documents or "document_evidence" in dimensions:
        skills.append("document_parsing")
    if "market" in dimensions:
        skills.append("market_data")
    if any(dim in dimensions for dim in ("profitability", "liquidity", "solvency", "r_and_d")):
        skills.append("financial_ratios")
    if "sentiment" in dimensions:
        skills.append("sentiment_analysis")
    if "compliance" in dimensions or "supply_chain" in dimensions:
        skills.append("compliance_review")
    skills.append("report_synthesis")
    return [skill for skill in skills if skill in SKILL_REGISTRY]


def _build_planner_notes(
    required_skills: list[str],
    missing_fields: list[str],
    document_contexts: list[dict[str, Any]],
    *,
    company_notes: list[str],
    time_notes: list[str],
    used_llm_structure: bool,
    retrieval_query: str,
    prefer_uploaded_only: bool = False,
    upload_mode_notes: list[str] | None = None,
) -> list[str]:
    notes = []
    notes.append(f"Selected {len(required_skills)} skills for the requested analysis.")
    notes.append(
        "Planner mode: rule layer + structured LLM extract."
        if used_llm_structure
        else "Planner mode: rule layer only (no LLM structure call)."
    )
    if document_contexts:
        notes.append(f"Detected {len(document_contexts)} uploaded document context(s).")
    notes.extend(company_notes)
    notes.extend(time_notes)
    notes.extend(upload_mode_notes or [])
    if prefer_uploaded_only:
        notes.append("Source mode: uploaded materials only (live fundamentals backfill disabled).")
    elif document_contexts:
        notes.append(
            "Source mode: hybrid (uploads first; SEC/Yahoo/sample may fill missing computable metrics)."
        )
    if retrieval_query:
        notes.append(f"retrieval_query={retrieval_query[:160]}")
    if missing_fields:
        notes.append(f"Missing fields were detected: {', '.join(missing_fields)}.")
    return notes
