"""Independent gold checks against source labels. FinAgentBench runs via evaluate_run."""

from __future__ import annotations

import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

# Ratio/percent display: compare against the operand-computed reference (or gold
# value if operands are absent). Minimum precision is 1 decimal percent or 3
# decimal ratio. A reported figure is accepted only if half-up rounding of the
# reference to that displayed precision matches. Not a per-task tolerance bump.
RATIO_DISPLAY = {
    "percent_min_decimals": 1,
    "ratio_min_decimals": 3,
    "rounding": "half_up",
}

from lumenfin.eval.document_tasks import ROOT

_REFUSE_HINTS = (
    "not available",
    "cannot determine",
    "cannot compute",
    "cannot answer",
    "cannot provide",
    "no computable",
    "incomplete",
    "not stated",
    "not in the uploaded",
    "does not provide",
    "do not provide",
    "uploaded files only",
    "data gap",
    "missing",
    "refuse",
    "insufficient",
    "do not invent",
    "cannot invent",
    "no operating income amount",
    "contains no",
)
_NUMBER_RE = re.compile(
    r"(?P<num>-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?)\s*(?P<unit>billion|million|percent|%|bn|mm)?",
    re.IGNORECASE,
)
_CITE_RE = re.compile(
    r"(?P<file>[\w.\-]+\.(?:pdf|txt|html|md))\s*(?:#p|p\.?\s*)(?P<page>\d+)",
    re.IGNORECASE,
)
_FILENAME_RE = re.compile(r"[\w.\-]+\.(?:pdf|txt|html|md)(?:#p\d+)?", re.IGNORECASE)
_GENERIC_CITE_NAMES = {"file.pdf", "excerpt.pdf", "x.pdf", "extract.pdf", "doc.pdf"}
_ENTITY_ALIASES = {
    "nvidia": ("nvidia", "nvda"),
    "microsoft": ("microsoft", "msft"),
    "apple": ("apple", "aapl"),
}
_METRIC_CUES = {
    "operating_income": ("operating income", "operating_income", "op. income"),
    "revenue": ("revenue", "net revenue"),
    "r_and_d": ("r&d", "r and d", "research and development", "r_and_d"),
    "r_and_d_intensity": ("r&d intensity", "rd intensity", "r_and_d_intensity"),
    "ebitda": ("ebitda",),
    "period_identity": ("fiscal year", "fy20", "operating income", "figures belong"),
    "operating_margin": ("operating margin",),
}
_NEGATION_PREFIXES = (
    "not ",
    "no ",
    "cannot ",
    "can't ",
    "isn't ",
    "is not ",
    "wasn't ",
    "was not ",
    "without ",
    "do not ",
    "don't ",
    "does not ",
    "doesn't ",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def _to_float(raw: str) -> float:
    return float(raw.replace(",", ""))


def _prose(text: str) -> str:
    """Drop filenames and bracket citations so fy2025 in a path is not a body claim."""
    stripped = _FILENAME_RE.sub(" ", str(text or ""))
    stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
    return _norm(stripped)


def _windows(text: str) -> list[str]:
    parts = re.split(r"(?<=[\n.!?|])\s+", str(text or "").strip())
    windows = [part.strip() for part in parts if part.strip()]
    return windows or [str(text or "")]


def _entity_in(text: str, entity: str) -> bool:
    if not entity:
        return True
    blob = _norm(text)
    key = _norm(entity)
    aliases = _ENTITY_ALIASES.get(key, (key,))
    return any(alias in blob for alias in aliases) or key in blob


def _metric_in(text: str, metric: str) -> bool:
    if not metric:
        return True
    blob = _norm(text)
    cues = _METRIC_CUES.get(str(metric), (str(metric).replace("_", " "),))
    return any(cue in blob for cue in cues)


def _period_tokens(period: str) -> tuple[str, ...]:
    period = str(period or "").strip()
    if not period or period.lower() in {"unknown", "null"}:
        return ()
    if not period.upper().startswith("FY"):
        return (period.lower(),)
    year = period[-4:]
    return (
        period.lower(),
        f"fy {year}",
        f"fiscal year {year}",
        f"fiscal year ended {year}",
        f"fiscal year ended: {year}",
    )


def _affirms_period(period: str, text: str) -> bool:
    blob = _prose(text)
    tokens = _period_tokens(period)
    if not tokens:
        return False
    if not any(token in blob for token in tokens):
        return False
    for token in tokens:
        if token not in blob:
            continue
        idx = blob.find(token)
        prefix = blob[max(0, idx - 18) : idx]
        if any(prefix.endswith(neg) or prefix.rstrip().endswith(neg.strip()) for neg in _NEGATION_PREFIXES):
            continue
        if re.search(rf"\bnot {re.escape(token)}\b", blob):
            continue
        return True
    return False


def _period_ok(fact: dict[str, Any], text: str) -> bool:
    period = str(fact.get("period") or "").strip()
    blob = _prose(text)
    if not period or period.lower() in {"unknown", "null"}:
        phrases = [str(item).lower() for item in fact.get("period_unspecified_phrases") or []]
        if not phrases:
            phrases = ["not stated", "not specified", "unspecified", "period unknown", "does not state", "no fiscal year"]
        return any(item in blob for item in phrases)
    return _affirms_period(period, text)


def _decimal_places(num_s: str) -> int:
    text = str(num_s or "").replace(",", "").strip()
    if "." not in text:
        return 0
    frac = text.split(".", 1)[1]
    return len(frac)


def _quantize(value: Decimal, places: int) -> Decimal:
    quantum = Decimal("1").scaleb(-places)
    return value.quantize(quantum, rounding=ROUND_HALF_UP)


def _ratio_reference(fact: dict[str, Any]) -> Decimal | None:
    ops = fact.get("operands")
    formula = str(fact.get("formula") or "")
    if isinstance(ops, dict) and "/" in formula:
        left, right = [part.strip() for part in formula.split("/", 1)]
        if left in ops and right in ops:
            denom = Decimal(str(ops[right]))
            if denom == 0:
                return None
            return Decimal(str(ops[left])) / denom
    if fact.get("value") is None:
        return None
    return Decimal(str(fact["value"]))


def _is_ratio_fact(fact: dict[str, Any], gold_unit: str) -> bool:
    if gold_unit.lower() in {"ratio", "percent", "%"}:
        return True
    ops = fact.get("operands")
    return isinstance(ops, dict) and "/" in str(fact.get("formula") or "")


def _ratio_display_match(fact: dict[str, Any], num_s: str, unit: str) -> bool:
    ref = _ratio_reference(fact)
    if ref is None:
        return False
    places = _decimal_places(num_s)
    reported = Decimal(str(num_s).replace(",", "").strip())
    unit = (unit or "").lower()
    if unit in {"%", "percent"}:
        if places < RATIO_DISPLAY["percent_min_decimals"]:
            return False
        return _quantize(ref * 100, places) == reported
    if places < RATIO_DISPLAY["ratio_min_decimals"]:
        return False
    return _quantize(ref, places) == reported


def _value_matches_fact(fact: dict[str, Any], converted: float, num_s: str, unit: str, gold_unit: str) -> bool:
    expected = float(fact["value"])
    tolerance = float(fact.get("abs_tolerance") or 0.0)
    if _is_ratio_fact(fact, gold_unit) and _ratio_display_match(fact, num_s, unit):
        return True
    return abs(converted - expected) <= max(tolerance, 1e-9)


def _to_canonical(value: float, unit: str, *, gold_unit: str) -> tuple[float | None, str]:
    unit = (unit or "").lower()
    gold_unit = (gold_unit or "").lower()
    if gold_unit in {"ratio", "percent", "%"}:
        if unit in {"%", "percent"}:
            return value / 100.0, "ratio"
        if unit == "" and 0 < value < 1.5:
            return value, "ratio"
        if unit == "" and 1.5 <= value <= 100:
            return None, "undetermined"
        return None, "skip"
    if gold_unit in {"billion", "bn"}:
        if unit in {"billion", "bn"}:
            return value, "billion"
        if unit in {"million", "mm"}:
            return value / 1000.0, "billion"
        return None, "undetermined"
    if gold_unit in {"million", "mm"}:
        if unit in {"million", "mm"}:
            return value, "million"
        if unit in {"billion", "bn"}:
            return value * 1000.0, "million"
        return None, "undetermined"
    if not unit:
        return None, "undetermined"
    return value, unit


def _looks_refuse(text: str, status: str) -> bool:
    if status in {"incomplete_data", "blocked_by_guardrail"}:
        return True
    blob = _norm(text)
    return any(hint in blob for hint in _REFUSE_HINTS)


def _positive_metric_claim(window: str, metric: str) -> bool:
    if not _metric_in(window, metric) and metric:
        return False
    blob = _norm(window)
    if _looks_refuse(window, ""):
        return False
    if any(blob.startswith(prefix) or f" {prefix}" in f" {blob}" for prefix in ("cannot ", "can't ", "do not ", "does not provide", "no amount")):
        if any(hint in blob for hint in _REFUSE_HINTS):
            return False
    return bool(_NUMBER_RE.search(window))


def _fact_hit(fact: dict[str, Any], text: str) -> tuple[bool, str | None]:
    phrases = fact.get("must_contain_any")
    if phrases:
        blob = _prose(text)
        if any(_norm(str(item)) in blob for item in phrases):
            return True, None
        return False, "missing_phrase"

    expected = fact.get("value")
    gold_unit = str(fact.get("unit") or "")
    if expected is None:
        period = str(fact.get("period") or "")
        ok = bool(period) and _period_ok(fact, text)
        return ok, None if ok else "missing_period"

    entity = str(fact.get("entity") or "")
    metric = str(fact.get("metric") or "")
    undetermined = False
    inherited_entity = ""
    inherited_period_ok = False
    for window in _windows(text):
        if any(_entity_in(window, name) for name in _ENTITY_ALIASES):
            inherited_entity = ""
            for name in _ENTITY_ALIASES:
                if _entity_in(window, name):
                    inherited_entity = name
                    break
            inherited_period_ok = False
        elif _entity_in(window, entity):
            inherited_entity = _norm(entity)
        period = str(fact.get("period") or "")
        if period.upper().startswith("FY") and _period_ok(fact, window):
            inherited_period_ok = True
        scoped = window
        if inherited_entity and not _entity_in(window, entity) and _norm(entity) == inherited_entity:
            scoped = f"{entity} {window}"
        if inherited_period_ok and period.upper().startswith("FY") and not _period_ok(fact, window):
            scoped = f"{period} {scoped}"
        if not _entity_in(scoped, entity):
            continue
        if metric and metric != "period_identity" and not _metric_in(scoped, metric):
            continue
        if _looks_refuse(window, "") and not _NUMBER_RE.search(window):
            continue
        for match in _NUMBER_RE.finditer(window):
            raw = _to_float(match.group("num"))
            unit = (match.group("unit") or "").lower()
            converted, status = _to_canonical(raw, unit, gold_unit=gold_unit or "billion")
            if converted is None:
                continue
            if status == "undetermined":
                undetermined = True
                continue
            if not _value_matches_fact(fact, converted, match.group("num"), unit, gold_unit):
                continue
            period = str(fact.get("period") or "")
            if period.lower() in {"unknown", "null"} or fact.get("period_must_be_unspecified"):
                if not _period_ok(fact, scoped) and not _period_ok(fact, text):
                    continue
            elif period.upper().startswith("FY"):
                if not _period_ok(fact, scoped):
                    continue
            return True, None
    if undetermined:
        return False, "needs_human_review"
    return False, "missing_fact"


def _risk_hit(point: dict[str, Any], text: str) -> bool:
    blob = _prose(text)
    needles = [str(item).lower() for item in point.get("must_cover") or []]
    return all(any(part in blob for part in needle.split()) or needle in blob for needle in needles)


def _looks_clarify(payload: dict[str, Any], text: str) -> bool:
    if payload.get("workflow_status") == "needs_clarification":
        return True
    if payload.get("missing_fields") or payload.get("clarification_questions"):
        return True
    blob = _norm(text)
    return any(
        word in blob
        for word in (
            "which company",
            "which fiscal",
            "which period",
            "which year",
            "please specify",
            "clarif",
        )
    )


def _page_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            return len(PdfReader(str(path)).pages)
        except Exception:
            return None
    if suffix in {".txt", ".html", ".md"}:
        return 1
    return None


def _citations(text: str, extra: list[str] | None) -> list[tuple[str, int]]:
    blob = str(text or "") + " " + " ".join(extra or [])
    found: list[tuple[str, int]] = []
    for match in _CITE_RE.finditer(blob):
        found.append((match.group("file").lower(), int(match.group("page"))))
    for match in re.finditer(
        r"(?:page|pages|p\.?)\s*(\d+)(?:\s*[-–]\s*\d+)?\s+(?:of\s+)?`?([\w.\-]+\.(?:pdf|txt|html|md))`?",
        blob,
        re.I,
    ):
        found.append((match.group(2).lower(), int(match.group(1))))
    for match in re.finditer(
        r"`?([\w.\-]+\.(?:pdf|txt|html|md))`?(?:\s+on)?\s+(?:page|pages|p\.?)\s*(\d+)",
        blob,
        re.I,
    ):
        found.append((match.group(1).lower(), int(match.group(2))))
    for match in re.finditer(
        r"`?([\w.\-]+\.(?:pdf|txt|html|md))`?\s*\((?:page|pages|p\.?)\s*(\d+)",
        blob,
        re.I,
    ):
        found.append((match.group(1).lower(), int(match.group(2))))
    return found


def _citation_ok(task: dict[str, Any], text: str, extra: list[str] | None) -> tuple[bool, str | None]:
    cites = _citations(text, extra)
    if not cites:
        return False, "missing_citation"
    allowed_docs = list(task.get("allowed_documents") or [])
    allowed_names = {Path(str(doc.get("path") or "")).name.lower() for doc in allowed_docs if doc.get("path")}
    page_limits: dict[str, int] = {}
    for doc in allowed_docs:
        rel = Path(str(doc.get("path") or ""))
        count = _page_count(ROOT / rel) if rel.parts else None
        if count:
            page_limits[rel.name.lower()] = count
    evidence_pages: set[int] = set()
    for fact in task.get("required_facts") or []:
        for group in fact.get("accepted_evidence_sets") or []:
            for item in group:
                try:
                    evidence_pages.add(int(item.get("page")))
                except (TypeError, ValueError):
                    continue
    for name, page in cites:
        if page < 1:
            continue
        if allowed_names and not _cite_name_allowed(name, allowed_names):
            continue
        limit = None
        if name in page_limits:
            limit = page_limits[name]
        elif len(page_limits) == 1:
            limit = next(iter(page_limits.values()))
        if limit is not None and page > limit:
            continue
        if not allowed_names and (name == "unrelated.pdf" or page > 50):
            continue
        return True, None
    return False, "invalid_citation"


def _cite_name_allowed(name: str, allowed_names: set[str]) -> bool:
    if name in allowed_names or name in _GENERIC_CITE_NAMES:
        return True
    stem = Path(name).stem.lower()
    if not stem:
        return False
    for allowed in allowed_names:
        allowed_stem = Path(allowed).stem.lower()
        if stem == allowed_stem or allowed_stem.startswith(stem + "_") or stem.startswith(allowed_stem):
            return True
    return False


def _forbidden_hit(banned: dict[str, Any], text: str) -> bool:
    entity = str(banned.get("entity") or "")
    period = banned.get("period")
    value = banned.get("value")
    metric = str(banned.get("metric") or "")
    if not entity and not period and value is None and not metric:
        return False
    for window in _windows(text):
        if entity and not _entity_in(window, entity):
            continue
        if metric and not _metric_in(window, metric) and value is None:
            continue
        if _looks_refuse(window, "") and value is None:
            continue
        if period and not _affirms_period(str(period), window):
            continue
        if value is None:
            if _positive_metric_claim(window, metric):
                return True
            continue
        gold_unit = str(banned.get("unit") or "billion")
        for match in _NUMBER_RE.finditer(window):
            converted, status = _to_canonical(_to_float(match.group("num")), match.group("unit") or "", gold_unit=gold_unit)
            if converted is None or status == "undetermined":
                continue
            if abs(converted - float(value)) <= 0.002 + 1e-9:
                if period and not _affirms_period(str(period), window):
                    continue
                if entity and not _entity_in(window, entity):
                    continue
                return True
    return False


def score_task(
    task: dict[str, Any],
    *,
    final_output: str,
    workflow_status: str = "completed",
    citations: list[str] | None = None,
    internal_metrics: list[dict[str, Any]] | None = None,
    finrun: dict[str, Any] | None = None,
    error: str | None = None,
    timed_out: bool = False,
    system: str | None = None,
) -> dict[str, Any]:
    """Score visible answers against gold. Empty output cannot succeed."""
    text = str(final_output or "")
    status = str(workflow_status or "completed")
    action = str(task.get("expected_action") or "answer")
    facts = list(task.get("required_facts") or [])
    risks = list(task.get("required_risk_points") or [])
    findings: list[str] = []
    undetermined = False
    if error or timed_out:
        findings.append("run_failed" if error else "timeout")
        return _pack(
            task,
            passed=False,
            action_ok=False,
            findings=findings,
            fact_hits=0,
            fact_total=max(len(facts), 1 if action == "answer" and not risks else 0),
            risk_hits=0,
            extra_unsupported=0,
            citation_ok=False,
            undetermined=False,
            error=error,
            final_output=text,
            finrun=finrun,
            system=system,
        )

    if not text.strip() and action == "answer" and status != "needs_clarification":
        findings.append("empty_output")
        return _pack(
            task,
            passed=False,
            action_ok=False,
            findings=findings,
            fact_hits=0,
            fact_total=max(len(facts), 1),
            risk_hits=0,
            extra_unsupported=0,
            citation_ok=False,
            undetermined=False,
            error=None,
            final_output=text,
            finrun=finrun,
            system=system,
        )

    action_ok = False
    clarify_ok = _looks_clarify({"workflow_status": status, **task}, text)
    if action == "refuse":
        asserted = any(
            _forbidden_hit({"value": item, "unit": "billion", "metric": "operating_income"}, text)
            for item in (task.get("refuse") or {}).get("must_not_assert_values") or []
        )
        action_ok = _looks_refuse(text, status) and not asserted
        if not action_ok:
            findings.append("expected_refuse")
    elif action == "clarify":
        action_ok = clarify_ok
        if not action_ok:
            findings.append("expected_clarify")
    elif action == "answer_or_clarify":
        action_ok = clarify_ok or not _looks_refuse(text, status)
        if not action_ok:
            findings.append("expected_answer_or_clarify")
    else:
        action_ok = not _looks_refuse(text, status) or bool(facts) or bool(risks)

    fact_hits = 0
    for fact in facts:
        ok, reason = _fact_hit(fact, text)
        if ok:
            fact_hits += 1
        else:
            if reason == "needs_human_review":
                undetermined = True
                findings.append("needs_human_review")
            findings.append(f"missing_fact:{fact.get('entity')}:{fact.get('metric')}:{fact.get('period')}")

    internal_gold_disagreements: list[str] = []
    if internal_metrics and str(system or "").startswith("b2"):
        for fact in facts:
            if fact.get("value") is None:
                continue
            for row in internal_metrics:
                if str(row.get("name") or row.get("metric")) != str(fact.get("metric")):
                    continue
                try:
                    inner = float(row.get("value"))
                except (TypeError, ValueError):
                    continue
                if abs(inner - float(fact["value"])) > float(fact.get("abs_tolerance") or 0) + 1e-9:
                    internal_gold_disagreements.append("internal_metric_disagrees_with_gold")

    risk_hits = sum(1 for point in risks if _risk_hit(point, text))
    if risks and risk_hits < len(risks):
        findings.append("missing_risk_point")

    extra_unsupported = 0
    for banned in task.get("forbidden_claims") or []:
        if _forbidden_hit(banned, text):
            extra_unsupported += 1
            findings.append("forbidden_claim")

    citation_ok = True
    needs_cite = (facts or risks) and action in {"answer", "answer_or_clarify"} and not (
        action == "answer_or_clarify" and clarify_ok and fact_hits < len(facts)
    )
    if needs_cite:
        citation_ok, cite_reason = _citation_ok(task, text, citations)
        if not citation_ok:
            findings.append(cite_reason or "missing_citation")

    fact_total = len(facts)
    if action == "answer" and not facts and not risks:
        fact_total = 1
        fact_hits = 0
        findings.append("answer_task_without_gold_facts")

    complete = True
    if action == "answer" and facts:
        complete = fact_hits == len(facts)
    if action == "answer" and risks:
        complete = complete and risk_hits == len(risks)
    if action in {"refuse", "clarify"}:
        complete = action_ok
    if action == "answer_or_clarify":
        complete = clarify_ok or (fact_hits == len(facts) and bool(facts))

    passed = (
        action_ok
        and complete
        and extra_unsupported == 0
        and (citation_ok or not needs_cite)
        and not undetermined
        and "empty_output" not in findings
    )
    if action == "answer" and facts and fact_hits < len(facts):
        passed = False
    if action == "refuse":
        passed = action_ok and extra_unsupported == 0
    if action == "clarify":
        passed = action_ok and extra_unsupported == 0
    if action == "answer_or_clarify":
        passed = (
            action_ok
            and complete
            and extra_unsupported == 0
            and (citation_ok or clarify_ok)
            and not undetermined
        )

    return _pack(
        task,
        passed=passed,
        action_ok=action_ok,
        findings=findings,
        fact_hits=fact_hits,
        fact_total=max(fact_total, len(facts)),
        risk_hits=risk_hits,
        extra_unsupported=extra_unsupported,
        citation_ok=citation_ok,
        undetermined=undetermined,
        error=None,
        final_output=text,
        finrun=finrun,
        system=system,
        internal_gold_disagreements=internal_gold_disagreements,
    )


def _pack(
    task: dict[str, Any],
    *,
    passed: bool,
    action_ok: bool,
    findings: list[str],
    fact_hits: int,
    fact_total: int,
    risk_hits: int,
    extra_unsupported: int,
    citation_ok: bool,
    undetermined: bool,
    error: str | None,
    final_output: str = "",
    finrun: dict[str, Any] | None = None,
    system: str | None = None,
    internal_gold_disagreements: list[str] | None = None,
    formal_pass: bool | None = None,
    scoring_eligibility: str | None = None,
) -> dict[str, Any]:
    from lumenfin.eval.contract_policy import SCORING_POLICY_VERSION
    from lumenfin.eval.contract_runner import acceptance_from_layers, evaluate_contract

    eligibility = scoring_eligibility or str(task.get("scoring_eligibility") or "candidate_gold")
    diagnostic_only = eligibility == "diagnostic_only"
    formal_eligible = bool(task.get("formal_accuracy_eligible"))
    formal = bool(formal_pass) if formal_pass is not None else False
    if formal_pass is None:
        formal = bool(passed) and formal_eligible and not diagnostic_only
    task_status = "undetermined" if undetermined else ("failed" if not passed else "passed")
    if error:
        task_status = "unavailable"
    contract = evaluate_contract(
        task,
        final_output=final_output,
        finrun=finrun,
        system=system,
    )
    internal = dict(contract.get("b2_internal_result") or {})
    extra_internal = list(internal_gold_disagreements or [])
    if extra_internal:
        internal_checks = list(internal.get("checks") or [])
        internal_checks.append(
            {
                "name": "internal_metric_vs_gold",
                "lane": "b2_internal",
                "status": "failed",
                "passed": False,
                "findings": extra_internal,
                "applicable_reason": "exported_metric_disagrees_with_gold",
            }
        )
        internal["checks"] = internal_checks
        internal["status"] = "failed"
        internal["passed"] = False
    contract_result = dict(contract.get("contract_result") or {})
    eval_acceptance = acceptance_from_layers(task_passed=passed, contract=contract)
    layer_a = {
        "available": str(contract.get("runner") or "") not in {"unavailable", "skipped_missing_finrun"},
        "status": contract_result.get("status"),
        "passed": bool(contract_result.get("passed")),
        "findings": [
            item.get("reason")
            for item in (contract.get("findings") or [])
            if item.get("lane") == "shared_export"
        ],
        "note": (
            "Replaced direct visible_supported_claims call. Layer A is export "
            "consistency via evaluate_run, not independent source gold. See contract_result."
        ),
        "replaced_by": "contract_result",
        "runner": contract.get("runner"),
    }
    return {
        "task_id": task.get("id"),
        "family": task.get("family"),
        "expected_action": task.get("expected_action"),
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "passed": passed,
        "diagnostic_pass": bool(passed),
        "formal_pass": formal,
        "scoring_eligibility": eligibility,
        "formal_accuracy_eligible": False,
        "eval_acceptance_v1": eval_acceptance,
        "eval_acceptance_v1_rule": (
            "task_result.passed and contract_result.status in {passed, not_applicable}; "
            "unavailable/error/undetermined/failed contract cannot pass; "
            "b2_internal_result is excluded; candidate gold is not formal accuracy"
        ),
        "action_ok": action_ok,
        "required_fact_coverage": {
            "hits": fact_hits,
            "total": fact_total,
            "denominator": "gold required_facts (0 for pure risk/refuse/clarify)",
        },
        "risk_points": {"hits": risk_hits, "total": len(task.get("required_risk_points") or [])},
        "citation_support": citation_ok,
        "extra_unsupported_claims": extra_unsupported,
        "findings": findings,
        "undetermined": undetermined,
        "error": error,
        "task_result": {
            "status": task_status,
            "passed": passed,
            "action_ok": action_ok,
            "findings": findings,
            "required_fact_coverage": {
                "hits": fact_hits,
                "total": fact_total,
            },
            "note": "Independent gold: source facts and whether the user task is complete.",
        },
        "contract_result": contract_result,
        "b2_internal_result": internal,
        "contract_findings": contract.get("findings") or [],
        "layer_a_finagentbench": layer_a,
        "note": (
            "Candidate gold diagnostic. diagnostic_pass remains gold-only and is not "
            "architecture-fair accuracy. eval_acceptance_v1 is a new development "
            "gate and does not change diagnostic_pass. formal_pass requires "
            "formal_accuracy_eligible on the task and is false for this development pilot."
        ),
    }
