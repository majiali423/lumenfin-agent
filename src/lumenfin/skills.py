from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SkillSpec:
    name: str
    description: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    owner_node: str


SKILL_REGISTRY: dict[str, SkillSpec] = {
    "company_identification": SkillSpec(
        name="company_identification",
        description="Extract target companies from the user query and uploaded document metadata.",
        inputs=("query", "document_contexts"),
        outputs=("companies", "target_symbols"),
        owner_node="query_planner",
    ),
    "document_parsing": SkillSpec(
        name="document_parsing",
        description="Parse uploaded PDF financial reports and extract text, company names, and metric hints.",
        inputs=("uploaded_files",),
        outputs=("document_contexts",),
        owner_node="service",
    ),
    "market_data": SkillSpec(
        name="market_data",
        description="Fetch external market snapshots when a ticker is available.",
        inputs=("companies", "target_symbols"),
        outputs=("market_snapshots",),
        owner_node="retrieval",
    ),
    "financial_ratios": SkillSpec(
        name="financial_ratios",
        description="Compute deterministic financial ratios with a restricted expression evaluator.",
        inputs=("retrieved_docs",),
        outputs=("financial_metrics",),
        owner_node="quant",
    ),
    "sentiment_analysis": SkillSpec(
        name="sentiment_analysis",
        description="Analyze management tone, strategic themes, and risk language from available text.",
        inputs=("earnings_call_quotes", "document_contexts"),
        outputs=("sentiment_analysis",),
        owner_node="psychologist",
    ),
    "compliance_review": SkillSpec(
        name="compliance_review",
        description="Check whether the final analysis contains required risk, data, and disclaimer coverage.",
        inputs=("financial_metrics", "sentiment_analysis", "final_report"),
        outputs=("compliance_findings", "compliance_summary"),
        owner_node="critic",
    ),
    "report_synthesis": SkillSpec(
        name="report_synthesis",
        description="Assemble the final report from structured evidence, metrics, risk scores, and audit output.",
        inputs=("query_plan", "financial_metrics", "risk_scores", "audit_log"),
        outputs=("final_report", "chart_data"),
        owner_node="synthesizer",
    ),
}


def get_skill_specs(skill_names: list[str]) -> list[dict[str, object]]:
    return [
        {
            "name": skill.name,
            "description": skill.description,
            "inputs": list(skill.inputs),
            "outputs": list(skill.outputs),
            "owner_node": skill.owner_node,
        }
        for name in skill_names
        if (skill := SKILL_REGISTRY.get(name)) is not None
    ]
