from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

GuardrailMode = Literal["sanitize", "block"]

INJECTION_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?", re.IGNORECASE),
        "critical",
        "ignore_previous_instructions",
    ),
    (
        re.compile(r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)", re.IGNORECASE),
        "critical",
        "disregard_previous_rules",
    ),
    (
        re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.IGNORECASE),
        "critical",
        "role_override",
    ),
    (
        re.compile(r"override\s+(?:safety|policy|guardrails?|system)", re.IGNORECASE),
        "critical",
        "policy_override",
    ),
    (
        re.compile(r"reveal\s+(?:the\s+)?(?:system|hidden)\s+prompt", re.IGNORECASE),
        "critical",
        "prompt_exfiltration",
    ),
    (
        re.compile(r"(?:^|\n)\s*system\s*:\s*", re.IGNORECASE | re.MULTILINE),
        "warning",
        "system_role_marker",
    ),
    (
        re.compile(r"(?:^|\n)\s*assistant\s*:\s*", re.IGNORECASE | re.MULTILINE),
        "warning",
        "assistant_role_marker",
    ),
    (
        re.compile(r"<\s*/?\s*(?:system|assistant|instruction)[^>]*>", re.IGNORECASE),
        "warning",
        "xml_instruction_tag",
    ),
    (
        re.compile(r"###\s*instruction", re.IGNORECASE),
        "warning",
        "markdown_instruction_header",
    ),
    (
        re.compile(
            r"\u5ffd\u7565(?:\u4e4b\u524d|\u6b64\u524d|\u4ee5\u4e0a|\u5148\u524d)"
            r"(?:\u7684)?(?:\u5168\u90e8|\u6240\u6709)?\u6307\u4ee4",
            re.IGNORECASE,
        ),
        "critical",
        "ignore_previous_instructions_zh",
    ),
    (
        re.compile(r"\u4f60\u73b0\u5728\u662f", re.IGNORECASE),
        "warning",
        "role_override_zh",
    ),
    (
        re.compile(r"\u65e0\u89c6(?:\u7cfb\u7edf|\u5b89\u5168|\u89c4\u5219)", re.IGNORECASE),
        "critical",
        "policy_override_zh",
    ),
]

REDACTION_TOKEN = "[REDACTED_INJECTION]"


@dataclass
class GuardrailFinding:
    document_id: str
    filename: str
    page: int | None
    pattern_id: str
    severity: str
    matched_text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "page": self.page,
            "pattern_id": self.pattern_id,
            "severity": self.severity,
            "matched_text": self.matched_text[:200],
        }


@dataclass
class GuardrailResult:
    allowed: bool
    mode: GuardrailMode
    findings: list[GuardrailFinding] = field(default_factory=list)
    sanitized_documents: list[dict[str, Any]] = field(default_factory=list)
    blocked_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "mode": self.mode,
            "findings": [finding.to_dict() for finding in self.findings],
            "blocked_reason": self.blocked_reason,
            "finding_count": len(self.findings),
            "critical_count": sum(1 for finding in self.findings if finding.severity == "critical"),
        }


def _redact_matches(text: str) -> tuple[str, list[tuple[str, str, str]]]:
    redacted = text
    local_hits: list[tuple[str, str, str]] = []
    for pattern, severity, pattern_id in INJECTION_PATTERNS:
        while True:
            match = pattern.search(redacted)
            if not match:
                break
            local_hits.append((pattern_id, severity, match.group(0)))
            redacted = (
                redacted[: match.start()]
                + REDACTION_TOKEN
                + redacted[match.end() :]
            )
    return redacted, local_hits


def scan_text(
    text: str,
    *,
    document_id: str,
    filename: str,
    page: int | None = None,
) -> list[GuardrailFinding]:
    findings: list[GuardrailFinding] = []
    for pattern, severity, pattern_id in INJECTION_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(
                GuardrailFinding(
                    document_id=document_id,
                    filename=filename,
                    page=page,
                    pattern_id=pattern_id,
                    severity=severity,
                    matched_text=match.group(0),
                )
            )
    return findings


def sanitize_document(document: dict[str, Any]) -> tuple[dict[str, Any], list[GuardrailFinding]]:
    sanitized = dict(document)
    findings: list[GuardrailFinding] = []
    document_id = str(document.get("document_id", "unknown"))
    filename = str(document.get("filename", "unknown"))

    pages = list(document.get("pages") or [])
    sanitized_pages: list[str] = []
    for page_number, page_text in enumerate(pages, start=1):
        redacted_text, hits = _redact_matches(page_text)
        sanitized_pages.append(redacted_text)
        for pattern_id, severity, matched_text in hits:
            findings.append(
                GuardrailFinding(
                    document_id=document_id,
                    filename=filename,
                    page=page_number,
                    pattern_id=pattern_id,
                    severity=severity,
                    matched_text=matched_text,
                )
            )

    if sanitized_pages:
        sanitized["pages"] = sanitized_pages
        sanitized["text"] = "\n".join(sanitized_pages).strip()
        sanitized["excerpt"] = sanitized["text"][:4000]
    elif document.get("text"):
        redacted_text, hits = _redact_matches(str(document["text"]))
        sanitized["text"] = redacted_text
        sanitized["excerpt"] = redacted_text[:4000]
        for pattern_id, severity, matched_text in hits:
            findings.append(
                GuardrailFinding(
                    document_id=document_id,
                    filename=filename,
                    page=None,
                    pattern_id=pattern_id,
                    severity=severity,
                    matched_text=matched_text,
                )
            )

    sanitized["guardrail_sanitized"] = bool(findings)
    return sanitized, findings


def guard_documents(
    documents: list[dict[str, Any]],
    *,
    mode: GuardrailMode = "sanitize",
) -> GuardrailResult:
    if not documents:
        return GuardrailResult(allowed=True, mode=mode, sanitized_documents=[])

    all_findings: list[GuardrailFinding] = []
    sanitized_documents: list[dict[str, Any]] = []
    for document in documents:
        sanitized, findings = sanitize_document(document)
        sanitized_documents.append(sanitized)
        all_findings.extend(findings)

    critical_count = sum(1 for finding in all_findings if finding.severity == "critical")
    if mode == "block" and critical_count > 0:
        return GuardrailResult(
            allowed=False,
            mode=mode,
            findings=all_findings,
            sanitized_documents=sanitized_documents,
            blocked_reason=(
                f"Blocked {critical_count} critical PDF prompt-injection pattern(s) "
                f"across {len(documents)} uploaded document(s)."
            ),
        )

    return GuardrailResult(
        allowed=True,
        mode=mode,
        findings=all_findings,
        sanitized_documents=sanitized_documents,
    )
