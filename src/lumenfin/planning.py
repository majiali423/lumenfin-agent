from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from .llm import BaseLLMClient
from .skills import SKILL_REGISTRY
from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .tools import KNOWN_ALIASES, extract_companies_from_query


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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
) -> QueryPlan:
    document_contexts = document_contexts or []
    normalized_query = " ".join(query.split())

    with ThreadPoolExecutor(max_workers=4) as executor:
        intent_future = executor.submit(_detect_intent, normalized_query, document_contexts)
        company_future = executor.submit(
            _detect_companies,
            normalized_query,
            document_contexts,
            llm_client,
        )
        dimension_future = executor.submit(_detect_dimensions, normalized_query, document_contexts)
        output_future = executor.submit(_detect_output_format, normalized_query)

        intent = intent_future.result()
        companies = company_future.result()
        dimensions = dimension_future.result()
        output_format = output_future.result()

    missing_fields = _detect_missing_fields(companies, document_contexts, normalized_query)
    clarification_questions = _build_clarification_questions(missing_fields)
    required_skills = _infer_required_skills(dimensions, has_documents=bool(document_contexts))
    planner_notes = _build_planner_notes(required_skills, missing_fields, document_contexts)

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
    )


def _detect_intent(query: str, document_contexts: list[dict[str, Any]]) -> str:
    lower = query.lower()
    if document_contexts or any(token in lower for token in ("pdf", "upload", "上传", "文件", "财报")):
        return "document_financial_diligence"
    if any(token in lower for token in ("compare", "versus", "vs", "对比", "比较")):
        return "comparative_financial_diligence"
    if any(token in lower for token in ("risk", "compliance", "audit", "风险", "合规", "审计")):
        return "risk_compliance_review"
    return "financial_diligence"


def _detect_companies(
    query: str,
    document_contexts: list[dict[str, Any]],
    llm_client: BaseLLMClient | None,
) -> list[str]:
    if not _has_explicit_company_signal(query, document_contexts):
        return []
    companies = extract_companies_from_query(query, document_contexts=document_contexts, llm_client=llm_client)
    for doc in document_contexts:
        for company in doc.get("detected_companies", []):
            if company not in companies:
                companies.append(company)
    return companies


def _has_explicit_company_signal(query: str, document_contexts: list[dict[str, Any]]) -> bool:
    lower = query.lower()
    if any(company.lower() in lower for company in SAMPLE_FINANCIAL_DATA):
        return True
    if any(alias in lower for alias in KNOWN_ALIASES):
        return True
    return any(doc.get("detected_companies") or doc.get("filename") for doc in document_contexts)


def _detect_dimensions(query: str, document_contexts: list[dict[str, Any]]) -> list[str]:
    lower = query.lower()
    dimensions = [
        dimension
        for dimension, keywords in DIMENSION_KEYWORDS.items()
        if any(keyword in lower for keyword in keywords)
    ]
    if document_contexts and "document_evidence" not in dimensions:
        dimensions.append("document_evidence")
    if not dimensions:
        dimensions = ["profitability", "r_and_d", "supply_chain", "sentiment", "compliance"]
    if "compliance" not in dimensions:
        dimensions.append("compliance")
    return dimensions


def _detect_output_format(query: str) -> str:
    lower = query.lower()
    if any(token in lower for token in ("table", "表格")):
        return "table_summary"
    if any(token in lower for token in ("brief", "summary", "摘要", "简版")):
        return "executive_summary"
    return "research_report"


def _detect_missing_fields(companies: list[str], document_contexts: list[dict[str, Any]], query: str) -> list[str]:
    missing = []
    if not companies and not document_contexts:
        missing.append("company")
    lower = query.lower()
    if not any(token in lower for token in ("2024", "2025", "2026", "fy", "财年", "年度", "year")):
        if not document_contexts:
            missing.append("time_range")
    return missing


def _build_clarification_questions(missing_fields: list[str]) -> list[str]:
    questions = []
    if "company" in missing_fields:
        questions.append("请明确要分析的公司名称（例如 Apple、Microsoft）。")
    if "time_range" in missing_fields:
        questions.append("请说明分析的时间范围或财年（例如 FY2025、2025 年报）。")
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
) -> list[str]:
    notes = []
    notes.append(f"Selected {len(required_skills)} skills for the requested analysis.")
    if document_contexts:
        notes.append(f"Detected {len(document_contexts)} uploaded document context(s).")
    if missing_fields:
        notes.append(f"Missing fields were detected: {', '.join(missing_fields)}.")
    return notes
