from __future__ import annotations

from typing import Any

from ..metrics_schema import get_fundamental
from ..parallel import map_in_parallel
from ..state import FinanceState
from ..tools import (
    build_coverage_matrix,
    calculate_derived_ratios,
    classify_quant_status,
    generate_scenario_analysis,
    is_partial_compare_gap,
    non_comparable_companies,
    resolve_safe_formula,
)
from .shared import _deterministic_peer_comparison, _single_company_peer_summary


class QuantitativeMixin:
    def _compute_company_quant(
        self,
        company: str,
        payload: dict[str, Any],
        state: FinanceState,
    ) -> dict[str, Any]:
        market = payload["market_data"]
        live_market = state.get("market_snapshots", {}).get(company, {})
        metrics: dict[str, float] = {}
        metric_confidence: dict[str, dict[str, Any]] = {}

        def set_confidence(metric_key: str, score: float, basis: str) -> None:
            metric_confidence[metric_key] = {
                "score": round(score, 2),
                "level": "High" if score >= 0.85 else ("Medium" if score >= 0.6 else "Low"),
                "basis": basis,
            }

        base_data: dict[str, float] = {}
        revenue = get_fundamental(market, "revenue")
        ebitda = get_fundamental(market, "ebitda")
        r_and_d = get_fundamental(market, "r_and_d")
        operating_income = get_fundamental(market, "operating_income")
        if revenue is not None:
            base_data["revenue"] = revenue
        if ebitda is not None:
            base_data["ebitda"] = ebitda
        if r_and_d is not None:
            base_data["r_and_d"] = r_and_d
        if operating_income is not None:
            base_data["operating_income"] = operating_income

        if len(base_data) >= 3:
            for formula, key in [
                ("ebitda / revenue", "ebitda_margin"),
                ("r_and_d / revenue", "r_and_d_intensity"),
                ("operating_income / revenue", "operating_margin"),
            ]:
                try:
                    if all(v in base_data for v in ["revenue"]):
                        if key == "r_and_d_intensity" and "r_and_d" not in base_data:
                            continue
                        if key == "operating_margin" and "operating_income" not in base_data:
                            continue
                        if key == "ebitda_margin" and "ebitda" not in base_data:
                            continue
                        metrics[key] = resolve_safe_formula(
                            formula,
                            base_data,
                            backend=self.tool_backend,
                        )
                        set_confidence(key, 0.95, "AST")
                except (KeyError, ValueError):
                    pass

        derived = calculate_derived_ratios(market)
        for key, value in derived.items():
            metrics.setdefault(key, value)
            if key not in metric_confidence:
                set_confidence(key, 0.72, "Derived")

        cap = live_market.get("market_cap")
        live_status = str(live_market.get("status") or "ok")
        live_conf = 0.8 if live_status == "ok" else (0.75 if live_status == "cached" else 0.55)
        live_basis = "LiveAPI" if live_status == "ok" else f"LiveAPI ({live_status})"
        if cap is not None:
            metrics["market_cap_billion"] = round(float(cap) / 1_000_000_000, 4)
            set_confidence("market_cap_billion", live_conf, live_basis)
        ret = live_market.get("monthly_return")
        if ret is not None:
            metrics["monthly_return"] = float(ret)
            set_confidence("monthly_return", live_conf, live_basis)
        cp = live_market.get("current_price")
        if cp is not None:
            metrics["current_price"] = float(cp)
            set_confidence("current_price", live_conf, live_basis)
        pe = live_market.get("trailing_pe")
        if pe is not None:
            metrics["pe_ratio"] = float(pe)
            set_confidence("pe_ratio", live_conf, live_basis)
        high = live_market.get("fifty_two_week_high")
        low = live_market.get("fifty_two_week_low")
        price = live_market.get("current_price")
        if all(v is not None for v in (high, low, price)) and float(high) != float(low):
            metrics["range_position"] = round((float(price) - float(low)) / (float(high) - float(low)), 4)
            set_confidence("range_position", live_conf * 0.97, live_basis)

        if not metrics:
            return {
                "company": company,
                "metrics": {},
                "quant_status": "uncomputable",
                "metric_confidence": {},
            }

        quant_status = classify_quant_status(metrics)
        return {
            "company": company,
            "metrics": metrics,
            "quant_status": quant_status,
            "scenario": generate_scenario_analysis(metrics, company) if quant_status == "ast_ok" else {},
            "metric_confidence": metric_confidence,
        }

    def quantitative_analyst(self, state: FinanceState) -> FinanceState:
        with self._track_step("quant") as timer:
            company_items = list(state["retrieved_docs"].items())
            quant_results = map_in_parallel(
                lambda item: self._compute_company_quant(item[0], item[1], state),
                company_items,
                max_workers=self.company_parallelism,
            )

            financial_metrics: dict[str, dict[str, float]] = {}
            scenario_analyses: dict[str, dict[str, Any]] = {}
            metric_confidence: dict[str, dict[str, dict[str, Any]]] = {}
            uncomputable: list[str] = []
            market_only: list[str] = []
            for result in quant_results:
                company = result["company"]
                status = str(result.get("quant_status") or classify_quant_status(result.get("metrics")))
                if status == "uncomputable":
                    uncomputable.append(company)
                    continue
                financial_metrics[company] = result.get("metrics") or {}
                scenario_analyses[company] = result.get("scenario") or {}
                metric_confidence[company] = result.get("metric_confidence") or {}
                if status == "market_only":
                    market_only.append(company)

            companies = list(state.get("companies") or [])
            coverage_matrix = build_coverage_matrix(companies, state.get("retrieved_docs") or {}, financial_metrics)
            partial_data_gap = is_partial_compare_gap(companies, coverage_matrix)
            skipped = non_comparable_companies(companies, coverage_matrix)

            comparable_metrics = {
                company: metrics
                for company, metrics in financial_metrics.items()
                if (coverage_matrix.get(company) or {}).get("comparable")
            }
            peer_comparison_text = ""
            if len(comparable_metrics) == 1:
                peer_comparison_text = _single_company_peer_summary(
                    next(iter(comparable_metrics))
                )
            elif comparable_metrics:
                peer_comparison_text = _deterministic_peer_comparison(comparable_metrics)
                if skipped:
                    peer_comparison_text += (
                        f" Non-comparable peers omitted: {', '.join(skipped)}."
                    )
            elif financial_metrics:
                peer_comparison_text = (
                    "Structured ratio comparison skipped: only market-level indicators were available "
                    f"for {', '.join(financial_metrics.keys())}."
                )
            else:
                peer_comparison_text = "No quantitative metrics were available for peer comparison."

            detail_parts = [
                f"Quantitative engine computed {sum(len(m) for m in financial_metrics.values())} metrics "
                f"across {len(financial_metrics)} companies (parallel fan-out)."
            ]
            if partial_data_gap:
                detail_parts.append(
                    f"Partial peer coverage: comparable={list(comparable_metrics.keys())}, "
                    f"non_comparable={skipped}."
                )
            if uncomputable:
                detail_parts.append(f"Uncomputable entities: {', '.join(uncomputable)}.")
            if market_only:
                detail_parts.append(f"Market-only entities: {', '.join(market_only)}.")
            detail_parts.append(f"Key insight: {peer_comparison_text[:120]}...")

            quant_status_label = "partial" if partial_data_gap else "ok"
            update: FinanceState = {
                "financial_metrics": financial_metrics,
                "metric_confidence": metric_confidence,
                "replan_reason": None,
                "partial_data_gap": partial_data_gap,
                "coverage_matrix": coverage_matrix,
                "non_comparable_companies": skipped,
                "degraded_mode": bool(state.get("degraded_mode")) or partial_data_gap,
                "tool_backend": self.tool_backend,
                "peer_comparison": {
                    "summary": peer_comparison_text,
                    "metrics": financial_metrics,
                    "scenarios": scenario_analyses,
                    "comparable_companies": list(comparable_metrics.keys()),
                    "non_comparable_companies": skipped,
                },
            }
            update.update(self._record("quant", quant_status_label, " ".join(detail_parts), state, timer.metrics()))
            self.session_memory.save({**state, **update})
            return update
