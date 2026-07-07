from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from .data.sample_financial_data import SAMPLE_FINANCIAL_DATA
from .market_data import DEFAULT_TICKER_MAP


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
    "amazon": "Amazon",
    "alphabet": "Alphabet",
    "google": "Alphabet",
    "meta": "Meta",
    "nvidia": "NVIDIA",
    "amd": "AMD",
}


def retrieve_company_payload(
    company: str,
    include_appendix: bool = False,
    document_contexts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a company payload from sample data, PDF documents, or both."""
    has_sample_data = company in SAMPLE_FINANCIAL_DATA
    if has_sample_data:
        payload = SAMPLE_FINANCIAL_DATA[company]
        result: dict[str, Any] = {
            "market_data": dict(payload["market_data"]),
            "supply_chain": dict(payload["supply_chain"]),
            "earnings_call_quotes": list(payload["earnings_call_quotes"]),
        }
        if include_appendix:
            result["appendix"] = dict(payload["appendix"])
        return result

    # For companies without sample data, extract from uploaded PDFs
    doc_contexts = document_contexts or []
    market_data: dict[str, float] = {}
    supply_chain_signals: list[str] = []
    earnings_quotes: list[str] = []

    for doc in doc_contexts:
        detected = doc.get("detected_companies", [])
        if company not in detected:
            continue
        text = doc.get("text", "")
        excerpt = doc.get("excerpt", "")[:3000]

        # Use PDF-extracted metrics as market data
        for key, value in doc.get("metric_hints", {}).items():
            if key == "revenue":
                market_data["revenue_2025"] = value
            elif key == "ebitda":
                market_data["ebitda_2025"] = value
            elif key == "r_and_d":
                market_data["r_and_d_2025"] = value

        # Extract narrative as earnings call quotes
        if excerpt:
            earnings_quotes.append(excerpt[:500])

        # Attempt to infer supply chain risk from text
        lowered_text = text.lower()
        risk_signals = []
        if any(w in lowered_text for w in ["supply chain risk", "供应链风险", "supply constraint", "logistics"]):
            risk_signals.append("PDF 文档中包含供应链相关讨论。")

        if risk_signals:
            supply_chain_signals = risk_signals

    if market_data:
        operating_income = market_data.get("ebitda_2025", 0) * 0.65
        market_data["operating_income_2025"] = round(operating_income, 1)

    result = {
        "market_data": market_data,
        "supply_chain": {
            "risk_level": "medium" if supply_chain_signals else "unknown",
            "signals": supply_chain_signals or ["PDF 文档中未检测到明确供应链信号。"],
        },
        "earnings_call_quotes": earnings_quotes or ["文档已上传，请基于 PDF 内容进行分析。"],
    }
    if include_appendix:
        result["appendix"] = {}
    return result


def _append_unique_company(companies: list[str], name: str) -> None:
    if name and name not in companies:
        companies.append(name)


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


def extract_companies_from_query(
    query: str,
    document_contexts: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
) -> list[str]:
    """Extract company names from query using sample data, PDF context, and LLM."""
    companies: list[str] = []
    lowered = query.lower()

    # 1. Check sample data for direct mentions
    for company in SAMPLE_FINANCIAL_DATA:
        if company.lower() in lowered:
            _append_unique_company(companies, company)

    # 2. Check known aliases
    for alias, name in KNOWN_ALIASES.items():
        if alias in lowered:
            _append_unique_company(companies, name)

    # 3. Collect companies detected in uploaded PDFs
    doc_contexts = document_contexts or []
    for doc in doc_contexts:
        for company in doc.get("detected_companies", []):
            _append_unique_company(companies, company)

    # 4. Merge LLM extraction so comparative queries do not stop at the first sample hit
    if llm_client:
        for company in _extract_companies_via_llm(query, llm_client):
            _append_unique_company(companies, company)

    if companies:
        return companies

    # 5. Fall back to PDF filename as company name hint
    for doc in doc_contexts:
        filename = doc.get("filename", "")
        if filename:
            name = Path(filename).stem
            if name and len(name) < 50:
                return [name]

    return []


def derive_target_symbols(companies: list[str], query: str) -> dict[str, str]:
    symbols = {company: DEFAULT_TICKER_MAP.get(company, company) for company in companies}
    explicit_tokens = re.findall(r"\b(?:ticker|symbol)\s*[:=]\s*([A-Z]{1,5})\b", query, flags=re.IGNORECASE)
    explicit_tokens.extend(re.findall(r"\(([A-Z]{1,5})\)", query))
    for token in explicit_tokens:
        for company in companies:
            if company not in symbols or symbols[company] == company:
                symbols[company] = token
                break
    return symbols


def summarize_document_context(document_contexts: list[dict[str, Any]], company: str) -> dict[str, Any]:
    related_docs = []
    metric_hints: dict[str, float] = {}
    for doc in document_contexts:
        if not doc.get("detected_companies") or company in doc.get("detected_companies", []):
            related_docs.append(
                {
                    "document_id": doc.get("document_id"),
                    "filename": doc.get("filename"),
                    "excerpt": doc.get("excerpt", "")[:1200],
                }
            )
            for metric_name, value in doc.get("metric_hints", {}).items():
                metric_hints.setdefault(metric_name, value)
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
    label = "bullish" if positive_hits >= caution_hits else "cautious"
    return {
        "label": label,
        "positive_hits": positive_hits,
        "caution_hits": caution_hits,
    }


def validate_report(report: str) -> list[str]:
    findings: list[str] = []
    if "风险免责声明" not in report:
        findings.append("缺少风险免责声明。")
    if "数据来源" not in report:
        findings.append("缺少数据来源标注。")
    return findings


def analyze_sentiment_deep(quotes: list[str], llm_client: Any | None = None) -> dict[str, Any]:
    """Deep sentiment analysis using LLM when available."""
    basic = analyze_sentiment(quotes)

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
            }
        except Exception:
            pass
    return basic


def calculate_derived_ratios(market_data: dict[str, float]) -> dict[str, float]:
    """Calculate additional financial ratios from available market data."""
    ratios: dict[str, float] = {}
    revenue = market_data.get("revenue_2025")
    ebitda = market_data.get("ebitda_2025")
    r_and_d = market_data.get("r_and_d_2025")
    op_income = market_data.get("operating_income_2025")

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
