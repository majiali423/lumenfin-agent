from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lumenfin.api.schemas import AnalyzeResponse
from lumenfin.claims.binding import _collect_evidence_pool
from lumenfin.claims.models import Claim, EvidenceRef, filter_verified
from lumenfin.structured_answer import (
    AllowedEvidence,
    CITATION_SOURCE_UNAVAILABLE,
    STRUCTURED_ANSWER_SCHEMA_VERSION,
    StructuredAnswerError,
    build_structured_answer_from_state,
    map_display_markers_to_chunk_ids,
    validate_structured_answer,
)


def _allowed(*chunk_ids: str, tenant: str = "tenant-a", session: str = "session-a") -> list[AllowedEvidence]:
    return [
        AllowedEvidence(
            chunk_id=chunk_id,
            tenant_id=tenant,
            session_id=session,
            verified=True,
        )
        for chunk_id in chunk_ids
    ]


class StructuredAnswerSchemaTests(unittest.TestCase):
    def test_single_citation_round_trip(self) -> None:
        answer = validate_structured_answer(
            {
                "answer": "Revenue was 10.",
                "citations": ["doc:p1:c0"],
                "structured_answer_schema_version": STRUCTURED_ANSWER_SCHEMA_VERSION,
            },
            allowed=_allowed("doc:p1:c0"),
            expected_tenant_id="tenant-a",
            expected_session_id="session-a",
        )
        self.assertEqual(answer.citations, ("doc:p1:c0",))
        self.assertEqual(answer.schema_version, "1.0")

    def test_multi_citation_preserves_first_seen_order_and_dedupes(self) -> None:
        answer = validate_structured_answer(
            {
                "answer": "Revenue grew.",
                "citations": ["a", "b", "a", "c"],
                "structured_answer_schema_version": "1.0",
            },
            allowed=_allowed("a", "b", "c"),
        )
        self.assertEqual(answer.citations, ("a", "b", "c"))

    def test_empty_and_non_string_citations_are_rejected(self) -> None:
        allowed = _allowed("a")
        with self.assertRaisesRegex(StructuredAnswerError, "non-empty strings"):
            validate_structured_answer(
                {"answer": "x", "citations": [""], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                require_citation_for_factual=False,
            )
        with self.assertRaisesRegex(StructuredAnswerError, "non-empty strings"):
            validate_structured_answer(
                {"answer": "x", "citations": [1], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                require_citation_for_factual=False,
            )
        with self.assertRaisesRegex(StructuredAnswerError, "string array"):
            validate_structured_answer(
                {"answer": "x", "citations": "a", "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                require_citation_for_factual=False,
            )

    def test_unknown_schema_version_is_rejected(self) -> None:
        with self.assertRaisesRegex(StructuredAnswerError, "unsupported"):
            validate_structured_answer(
                {
                    "answer": "ok",
                    "citations": ["a"],
                    "structured_answer_schema_version": "9.9",
                },
                allowed=_allowed("a"),
            )


class StructuredAnswerBindingTests(unittest.TestCase):
    def test_verified_chunk_passes_unknown_and_unverified_fail(self) -> None:
        allowed = [
            AllowedEvidence(chunk_id="good", tenant_id="t", session_id="s", verified=True),
            AllowedEvidence(chunk_id="raw", tenant_id="t", session_id="s", verified=False),
        ]
        validate_structured_answer(
            {"answer": "ok", "citations": ["good"], "structured_answer_schema_version": "1.0"},
            allowed=allowed,
            expected_tenant_id="t",
            expected_session_id="s",
            require_citation_for_factual=False,
        )
        with self.assertRaisesRegex(StructuredAnswerError, "unknown"):
            validate_structured_answer(
                {"answer": "ok", "citations": ["missing"], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                require_citation_for_factual=False,
            )
        with self.assertRaisesRegex(StructuredAnswerError, "unverified"):
            validate_structured_answer(
                {"answer": "ok", "citations": ["raw"], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                require_citation_for_factual=False,
            )

    def test_cross_tenant_session_and_stale_repair_fail(self) -> None:
        allowed = [
            AllowedEvidence(chunk_id="other-tenant", tenant_id="t2", session_id="s", verified=True),
            AllowedEvidence(chunk_id="other-session", tenant_id="t", session_id="s2", verified=True),
            AllowedEvidence(chunk_id="old", tenant_id="t", session_id="s", verified=True, stale=True),
        ]
        with self.assertRaisesRegex(StructuredAnswerError, "tenant"):
            validate_structured_answer(
                {"answer": "ok", "citations": ["other-tenant"], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                expected_tenant_id="t",
                expected_session_id="s",
                require_citation_for_factual=False,
            )
        with self.assertRaisesRegex(StructuredAnswerError, "session"):
            validate_structured_answer(
                {"answer": "ok", "citations": ["other-session"], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                expected_tenant_id="t",
                expected_session_id="s",
                require_citation_for_factual=False,
            )
        with self.assertRaisesRegex(StructuredAnswerError, "stale"):
            validate_structured_answer(
                {"answer": "ok", "citations": ["old"], "structured_answer_schema_version": "1.0"},
                allowed=allowed,
                expected_tenant_id="t",
                expected_session_id="s",
                require_citation_for_factual=False,
            )

    def test_rag_pool_preserves_chunk_id(self) -> None:
        pool = _collect_evidence_pool(
            {
                "thread_id": "session-a",
                "rag_tenant_id": "tenant-a",
                "rag_evidence": {
                    "Apple": [
                        {
                            "chunk_id": "apple:p1:c0",
                            "citation": "10k.pdf#p1",
                            "text": "Apple revenue was 412.",
                            "tenant_id": "tenant-a",
                        }
                    ]
                },
            },
            "Apple",
        )
        self.assertEqual(pool[0].chunk_id, "apple:p1:c0")
        self.assertEqual(pool[0].tenant_id, "tenant-a")
        self.assertEqual(pool[0].session_id, "session-a")


class StructuredAnswerPolicyTests(unittest.TestCase):
    def test_factual_answer_without_citation_fails_when_rag_evidence_exists(self) -> None:
        with self.assertRaisesRegex(StructuredAnswerError, "at least one citation"):
            validate_structured_answer(
                {
                    "answer": "EBITDA margin was 34%.",
                    "citations": [],
                    "structured_answer_schema_version": "1.0",
                    "workflow_status": "completed",
                },
                allowed=_allowed("doc:p1:c0"),
            )

    def test_incomplete_data_without_citation_is_allowed(self) -> None:
        answer = validate_structured_answer(
            {
                "answer": "No computable structured fundamentals were available.",
                "citations": [],
                "structured_answer_schema_version": "1.0",
                "workflow_status": "incomplete_data",
                "citation_source": CITATION_SOURCE_UNAVAILABLE,
            },
            allowed=_allowed("doc:p1:c0"),
            require_citation_for_factual=False,
        )
        self.assertEqual(answer.citations, ())

    def test_incomplete_data_cannot_carry_unsupported_ratio(self) -> None:
        with self.assertRaisesRegex(StructuredAnswerError, "unsupported financial ratios"):
            validate_structured_answer(
                {
                    "answer": "EBITDA margin was 34% despite missing filings.",
                    "citations": [],
                    "structured_answer_schema_version": "1.0",
                    "workflow_status": "incomplete_data",
                    "citation_source": CITATION_SOURCE_UNAVAILABLE,
                },
                allowed=[],
                require_citation_for_factual=False,
            )

    def test_display_markers_need_an_explicit_map_and_do_not_guess(self) -> None:
        mapped = map_display_markers_to_chunk_ids([1, 2, 1], {1: "a", 2: "b"})
        self.assertEqual(mapped, ["a", "b"])
        with self.assertRaisesRegex(StructuredAnswerError, "not in the evidence map"):
            map_display_markers_to_chunk_ids([9], {1: "a"})

    def test_builder_uses_verified_chunk_ids_only(self) -> None:
        state = {
            "thread_id": "session-a",
            "rag_tenant_id": "tenant-a",
            "workflow_status": "completed",
            "final_report": "Revenue was 412 billion.",
            "rag_evidence": {
                "Apple": [
                    {
                        "chunk_id": "apple:p1:c0",
                        "tenant_id": "tenant-a",
                        "session_id": "session-a",
                        "text": "Revenue was 412 billion.",
                    },
                    {
                        "chunk_id": "apple:p2:c0",
                        "tenant_id": "tenant-a",
                        "session_id": "session-a",
                        "text": "Unbound passage.",
                    },
                ]
            },
            "claims": [
                Claim(
                    claim_id="c1",
                    entity="Apple",
                    claim_type="numeric",
                    statement="Apple revenue was 412.",
                    value=412.0,
                    verification="verified",
                    evidence_refs=[
                        EvidenceRef(
                            evidence_id="ev1",
                            entity="Apple",
                            citation="10k.pdf#p1",
                            source_type="rag",
                            text="Revenue was 412 billion.",
                            chunk_id="apple:p1:c0",
                            tenant_id="tenant-a",
                            session_id="session-a",
                        )
                    ],
                ).to_dict()
            ],
            "verified_claims": [],
        }
        state["verified_claims"] = [item for item in state["claims"]]
        payload = build_structured_answer_from_state(state)
        self.assertEqual(payload.citations, ("apple:p1:c0",))
        self.assertNotIn("apple:p2:c0", payload.citations)

    def test_filter_verified_still_requires_evidence_refs(self) -> None:
        claims = [
            Claim(
                claim_id="empty",
                entity="Apple",
                claim_type="numeric",
                statement="x",
                verification="verified",
            )
        ]
        self.assertEqual(filter_verified(claims), [])

    def test_api_schema_keeps_final_report_and_adds_optional_citation_fields(self) -> None:
        payload = AnalyzeResponse(
            thread_id="t",
            llm_backend="local-fallback",
            final_report="Revenue was 10.",
            audit_log=[],
            artifacts={},
            state={"workflow_status": "completed"},
            answer="Revenue was 10.",
            citations=["doc:p1:c0"],
            structured_answer_schema_version="1.0",
        )
        dumped = payload.model_dump()
        self.assertEqual(dumped["final_report"], "Revenue was 10.")
        self.assertEqual(dumped["answer"], "Revenue was 10.")
        self.assertEqual(dumped["citations"], ["doc:p1:c0"])
        legacy = AnalyzeResponse(
            thread_id="t",
            llm_backend="local-fallback",
            final_report="legacy only",
            audit_log=[],
            artifacts={},
            state={},
        )
        self.assertEqual(legacy.final_report, "legacy only")
        self.assertEqual(legacy.citations, [])


if __name__ == "__main__":
    unittest.main()
