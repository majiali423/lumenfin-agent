from __future__ import annotations

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

from lumenfin.eval.document_tasks import (
    CATALOG_PATH,
    DOCUMENT_HASH_POLICY_ID,
    ROOT,
    evidence_quote_issues,
    freeze_readiness,
    load_catalog,
    sha256_bytes,
    sha256_document,
    sha256_file,
    tasks_for,
)
from lumenfin.eval.gold_evaluator import CITATION_SUPPORT_POLICY_VERSION, score_task


class CatalogTestCase(unittest.TestCase):
    def test_pilot_has_24_tasks_with_honest_review_status(self) -> None:
        catalog = load_catalog()
        self.assertEqual(catalog["dataset_id"], "lumenfin_document_tasks_v1")
        self.assertEqual(catalog["version"], "2026-09-14.pilot24")
        self.assertEqual(catalog["document_hash_policy"]["id"], "canonical_text_eol_lf.v1")
        pilots = tasks_for(pilot_only=True)
        self.assertEqual(len(pilots), 24)
        families = {row["family"] for row in pilots}
        self.assertEqual(len(families), 6)
        for family in families:
            self.assertEqual(sum(1 for row in pilots if row["family"] == family), 4)
        self.assertTrue(all(row["split"] == "dev" for row in pilots))
        allowed = {"source_quote_checked", "needs_human_review"}
        self.assertTrue(all(row["review_status"] in allowed for row in pilots))
        self.assertFalse(any(row["review_status"] in {"author_verified", "frozen", "human_reviewed"} for row in pilots))
        human = [row["id"] for row in pilots if row["review_status"] == "needs_human_review"]
        self.assertEqual(human, ["dt-p23-clarify-company"])
        self.assertEqual(catalog.get("formal_accuracy_eligible"), False)
        groups = {item["id"]: item["split"] for item in catalog["groups"]}
        for row in pilots:
            self.assertEqual(groups[row["source_group"]], "dev")
            for peer in row.get("comparison_groups") or []:
                self.assertEqual(groups[peer], "dev")
        consumed = " ".join(catalog["consumed_sets_not_reused_as_unseen"])
        self.assertIn("confirmation-50", consumed)
        self.assertIn("public_holdout", consumed)
        blocked = [item["id"] for item in catalog["groups"] if item.get("status") == "blocked_needs_source_extract"]
        self.assertEqual(len(blocked), 6)

    def test_pilot_quotes_match_stated_pages(self) -> None:
        issues = []
        for task in tasks_for(pilot_only=True):
            issues.extend(evidence_quote_issues(task))
        self.assertEqual(issues, [])

    def test_freeze_readiness_is_data_not_exception(self) -> None:
        payload = freeze_readiness()
        self.assertFalse(payload["frozen"])
        self.assertFalse(payload["program_exception"])
        self.assertEqual(payload["exit_code"], 2)
        ids = {item["id"] for item in payload["blockers"]}
        self.assertEqual(
            ids,
            {
                "no_frozen_test_gold",
                "issuer_groups_missing_extracts",
                "no_independent_human_review",
                "pilot_not_a_test_freeze",
            },
        )
        for item in payload["blockers"]:
            self.assertTrue(item["unlock"])


class GoldEvaluatorTestCase(unittest.TestCase):
    def _fact_task(self) -> dict:
        return {
            "id": "unit-oi",
            "family": "financial_fact",
            "expected_action": "answer",
            "required_facts": [{
                "entity": "NVIDIA",
                "metric": "operating_income",
                "period": "FY2025",
                "value": 81.453,
                "unit": "billion",
                "abs_tolerance": 0.002,
            }],
            "allowed_documents": [{"path": "excerpt.pdf", "document_id": "excerpt"}],
            "forbidden_claims": [],
        }

    def test_correct_and_equivalent_billion(self) -> None:
        task = self._fact_task()
        ok = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1].",
        )
        self.assertTrue(ok["passed"], ok["findings"])
        also = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81,453 million USD [excerpt.pdf#p1].",
        )
        self.assertTrue(also["passed"], also["findings"])

    def test_wrong_amount_unit_period_entity_and_missing_fact(self) -> None:
        task = self._fact_task()
        self.assertFalse(score_task(task, final_output="NVIDIA FY2025 operating income was 72.4 billion USD [x.pdf#p1].")["passed"])
        self.assertFalse(score_task(task, final_output="Microsoft FY2025 operating income was 81.453 billion USD [x.pdf#p1].")["passed"])
        self.assertFalse(score_task(task, final_output="NVIDIA FY2024 operating income was 81.453 billion USD [x.pdf#p1].")["passed"])
        self.assertFalse(score_task(task, final_output="")["passed"])

    def test_unit_and_entity_cannot_be_assembled_across_sentences(self) -> None:
        task = self._fact_task()
        million = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 million USD [excerpt.pdf#p1].",
        )
        self.assertFalse(million["passed"], million["findings"])
        crossed = score_task(
            task,
            final_output=(
                "NVIDIA operating income is 7 billion USD. "
                "Microsoft revenue is 81.453 billion USD [excerpt.pdf#p1]."
            ),
        )
        self.assertFalse(crossed["passed"], crossed["findings"])
        bogus_cite = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [unrelated.pdf#p999].",
        )
        self.assertFalse(bogus_cite["passed"], bogus_cite["findings"])
        self.assertIn("invalid_citation", bogus_cite["findings"])

    def test_correct_refuse_is_not_a_forbidden_assertion(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p21-narrative-refuse-oi")
        scored = score_task(
            task,
            final_output=(
                "The retrieved passage does not provide NVIDIA FY2025 operating income. "
                "The only uploaded excerpt (nvda_narrative_only.txt#p1) explicitly states "
                "that it contains no operating income amount, so I cannot answer with a number."
            ),
        )
        self.assertTrue(scored["passed"], scored["findings"])
        self.assertTrue(scored["diagnostic_pass"])
        self.assertFalse(scored["formal_pass"])

    def test_candidate_gold_pass_is_not_formal_pass(self) -> None:
        scored = score_task(
            self._fact_task(),
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1].",
        )
        self.assertTrue(scored["passed"])
        self.assertTrue(scored["diagnostic_pass"])
        self.assertFalse(scored["formal_pass"])
        self.assertFalse(scored["formal_accuracy_eligible"])

    def test_percent_margin_and_prose_citation(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p05-nvda-op-margin")
        scored = score_task(
            task,
            final_output=(
                "NVIDIA FY2025 operating margin was 62.42% as stated on pages 1-5 of "
                "`nvda_fy2025_10k_excerpt.pdf`."
            ),
        )
        self.assertTrue(scored["passed"], scored["findings"])

    def test_ratio_display_precision_from_operands(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p06-msft-op-margin")
        cite = "msft_fy2024_10k_long_excerpt.pdf#p1"
        one_decimal = score_task(
            task,
            final_output=f"Microsoft FY2024 operating margin was 44.6% [{cite}].",
        )
        self.assertTrue(one_decimal["passed"], one_decimal["findings"])
        two_decimal = score_task(
            task,
            final_output=f"Microsoft FY2024 operating margin was 44.64% [{cite}].",
        )
        self.assertTrue(two_decimal["passed"], two_decimal["findings"])
        ratio = score_task(
            task,
            final_output=f"Microsoft FY2024 operating margin was 0.446 [{cite}].",
        )
        self.assertTrue(ratio["passed"], ratio["findings"])
        nvda = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p05-nvda-op-margin")
        nvda_cite = "nvda_fy2025_10k_excerpt.pdf#p1"
        self.assertTrue(
            score_task(nvda, final_output=f"NVIDIA FY2025 operating margin was 62.4% [{nvda_cite}].")["passed"]
        )

    def test_ratio_display_rejects_coarse_or_wrong_values(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p06-msft-op-margin")
        cite = "msft_fy2024_10k_long_excerpt.pdf#p1"
        for text in (
            f"Microsoft FY2024 operating margin was 45% [{cite}].",
            f"Microsoft FY2024 operating margin was 44% [{cite}].",
            f"Microsoft FY2024 operating margin was 40% [{cite}].",
            f"Microsoft FY2024 operating margin was 44.0% [{cite}].",
            f"Microsoft FY2024 operating margin was 0.45 [{cite}].",
        ):
            scored = score_task(task, final_output=text)
            self.assertFalse(scored["passed"], text)
        wrong_entity = score_task(
            task,
            final_output="NVIDIA FY2024 operating margin was 44.6% [file.pdf#p1].",
        )
        self.assertFalse(wrong_entity["passed"])
        wrong_period = score_task(
            task,
            final_output=f"Microsoft FY2025 operating margin was 44.6% [{cite}].",
        )
        self.assertFalse(wrong_period["passed"])
        dollar = self._fact_task()
        self.assertFalse(
            score_task(
                dollar,
                final_output="NVIDIA FY2025 operating income was 81.5 billion USD [excerpt.pdf#p1].",
            )["passed"]
        )

    def test_margin_can_use_prior_sentence_entity(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p05-nvda-op-margin")
        scored = score_task(
            task,
            final_output=(
                "The passages do not state NVIDIA FY2025 operating margin directly.\n"
                "Using those figures, operating margin = 81,453 / 130,497 ≈ 62.42%.\n"
                "Source: nvda_fy2025_10k_excerpt.pdf#p1."
            ),
        )
        self.assertTrue(scored["passed"], scored["findings"])

    def test_internal_and_visible_same_wrong_value_still_fails_gold(self) -> None:
        task = self._fact_task()
        scored = score_task(
            task,
            final_output="NVIDIA FY2025 operating income is 99.0 billion USD [x.pdf#p1].",
            internal_metrics=[{"name": "operating_income", "value": 99.0, "entity": "NVIDIA"}],
        )
        self.assertFalse(scored["passed"])
        self.assertTrue(
            any("internal_metric_disagrees_with_gold" in item or "missing_fact" in item for item in scored["findings"])
        )

    def test_refuse_and_ungrounded_answer(self) -> None:
        refuse = {
            "id": "unit-refuse",
            "family": "refuse_or_clarify",
            "expected_action": "refuse",
            "required_facts": [],
            "refuse": {"must_not_assert_values": [81.453, 72.4]},
            "forbidden_claims": [{"entity": "NVIDIA", "metric": "operating_income", "value": 81.453}],
        }
        self.assertTrue(score_task(refuse, final_output="Cannot determine operating income from the uploaded narrative.")["passed"])
        self.assertFalse(score_task(refuse, final_output="NVIDIA operating income is 81.453 billion.")["passed"])

    def test_empty_cannot_score_full_marks(self) -> None:
        scored = score_task(self._fact_task(), final_output="   ")
        self.assertFalse(scored["passed"])
        self.assertIn("empty_output", scored["findings"])


class AmbiguousTaskRulesTestCase(unittest.TestCase):
    def test_p10_requires_amount_and_unspecified_period(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p10-nvda-unspecified-period")
        ok = score_task(
            task,
            final_output="NVIDIA operating income is 32.972 billion USD; the fiscal year is not stated next to that figure [nvda.pdf#p2].",
        )
        self.assertTrue(ok["passed"], ok["findings"])
        self.assertFalse(ok["formal_accuracy_eligible"])
        self.assertFalse(
            score_task(task, final_output="NVIDIA FY2025 operating income was 32.972 billion USD [nvda.pdf#p2].")["passed"]
        )
        self.assertFalse(
            score_task(task, final_output="NVIDIA operating income is 32.972 billion USD [nvda.pdf#p2].")["passed"]
        )

    def test_p11_yes_no_needs_fy2024_binding(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p11-nvda-wrong-year-reject")
        ok = score_task(
            task,
            final_output="No. 32.972 billion USD is NVIDIA FY2024 operating income on page 2, not FY2025 [file.pdf#p2].",
        )
        self.assertTrue(ok["passed"], ok["findings"])
        self.assertFalse(score_task(task, final_output="Yes.")["passed"])
        self.assertFalse(
            score_task(task, final_output="Yes, NVIDIA FY2025 operating income is 32.972 billion USD [file.pdf#p2].")["passed"]
        )
        self.assertFalse(score_task(task, final_output="No. [file.pdf#p2]")["passed"])

    def test_p12_does_not_pass_on_bare_2024(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p12-msft-period-identity")
        ok = score_task(
            task,
            final_output="The operating income figures belong to FY2024 (fiscal year ended 2024-06-30) [msft.pdf#p1].",
        )
        self.assertTrue(ok["passed"], ok["findings"])
        self.assertFalse(score_task(task, final_output="The file was filed in 2024 [msft.pdf#p1].")["passed"])
        self.assertFalse(
            score_task(task, final_output="Operating income is 109,433 million USD [msft.pdf#p1].")["passed"]
        )

    def test_p23_is_diagnostic_only(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p23-clarify-company")
        diag = score_task(task, final_output="Which company and which fiscal year do you mean by last year?")
        self.assertTrue(diag["passed"], diag["findings"])
        self.assertFalse(diag["formal_pass"])
        self.assertEqual(diag["scoring_eligibility"], "diagnostic_only")
        answered = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [nvda.pdf#p1].",
        )
        self.assertFalse(answered["passed"])
        self.assertFalse(answered["formal_pass"])

    def test_p24_answer_or_clarify(self) -> None:
        task = next(item for item in tasks_for(pilot_only=True) if item["id"] == "dt-p24-clarify-period")
        clarify = score_task(task, final_output="Which fiscal year? Cover mentions FY2025 and page 2 states FY2024.")
        self.assertTrue(clarify["passed"], clarify["findings"])
        answer = score_task(
            task,
            final_output="NVIDIA FY2024 operating income was 32.972 billion USD [file.pdf#p2].",
        )
        self.assertTrue(answer["passed"], answer["findings"])
        self.assertFalse(
            score_task(task, final_output="NVIDIA FY2025 operating income was 32.972 billion USD [file.pdf#p2].")["passed"]
        )


class DocumentHashPolicyTestCase(unittest.TestCase):
    HTML_REL = "tests/fixtures/sec/minimal/aapl_fy2024_10k_extract.html"
    PDF_REL = "tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf"
    PRIOR_HTML_CRLF = "73d60feeffe30932ea10bbc5a7f99e575ca325860f32d3631255306100d2f456"

    def _git_blob(self, rel: str) -> bytes:
        return subprocess.check_output(
            ["git", "cat-file", "blob", f"HEAD:{rel}"],
            cwd=str(ROOT),
        )

    def test_lf_and_crlf_text_share_git_blob_identity(self) -> None:
        blob = self._git_blob(self.HTML_REL)
        self.assertNotIn(b"\r\n", blob)
        crlf = blob.replace(b"\n", b"\r\n")
        self.assertNotEqual(hashlib.sha256(blob).hexdigest(), hashlib.sha256(crlf).hexdigest())
        self.assertEqual(sha256_bytes(blob, suffix=".html"), sha256_bytes(crlf, suffix=".html"))
        catalog = load_catalog()
        expected = None
        for task in catalog["tasks"]:
            for doc in task.get("allowed_documents") or []:
                if doc.get("path") == self.HTML_REL:
                    expected = doc["sha256"]
                    break
        self.assertEqual(expected, sha256_bytes(blob, suffix=".html"))
        self.assertEqual(catalog["document_hash_policy"]["id"], DOCUMENT_HASH_POLICY_ID)
        self.assertEqual(catalog["document_hash_policy"]["prior_text_sha256"][self.HTML_REL], self.PRIOR_HTML_CRLF)
        self.assertNotEqual(expected, self.PRIOR_HTML_CRLF)
        with tempfile.TemporaryDirectory() as tmp:
            lf_path = Path(tmp) / "a.html"
            crlf_path = Path(tmp) / "b.html"
            lf_path.write_bytes(blob)
            crlf_path.write_bytes(crlf)
            self.assertEqual(sha256_document(lf_path), sha256_document(crlf_path))
            self.assertEqual(sha256_document(lf_path), expected)

    def test_real_content_change_changes_text_hash(self) -> None:
        blob = self._git_blob(self.HTML_REL)
        mutated = blob.replace(b"123,216", b"999,999", 1)
        self.assertNotEqual(blob, mutated)
        self.assertNotEqual(sha256_bytes(blob, suffix=".html"), sha256_bytes(mutated, suffix=".html"))

    def test_pdf_keeps_raw_bytes(self) -> None:
        pdf = ROOT / self.PDF_REL
        blob = self._git_blob(self.PDF_REL)
        self.assertEqual(sha256_document(pdf), sha256_file(pdf))
        self.assertEqual(sha256_document(pdf), hashlib.sha256(blob).hexdigest())
        self.assertNotEqual(sha256_bytes(blob, suffix=".html"), hashlib.sha256(blob).hexdigest())

    def test_catalog_load_matches_git_checkout_bytes(self) -> None:
        catalog = load_catalog()
        self.assertEqual(CATALOG_PATH.name, "lumenfin_document_tasks_v1.json")
        by_path: dict[str, str] = {}
        for task in catalog["tasks"]:
            for doc in task.get("allowed_documents") or []:
                by_path[str(doc["path"])] = str(doc["sha256"])
        for rel, digest in by_path.items():
            blob = self._git_blob(rel)
            suffix = Path(rel).suffix
            self.assertEqual(sha256_bytes(blob, suffix=suffix), digest, rel)


class CitationSupportTestCase(unittest.TestCase):
    def _two_issuer_task(self) -> dict:
        return {
            "id": "unit-two-docs",
            "family": "comparison_scope",
            "expected_action": "answer",
            "required_facts": [
                {
                    "entity": "NVIDIA",
                    "metric": "operating_income",
                    "period": "FY2025",
                    "value": 81.453,
                    "unit": "billion",
                    "abs_tolerance": 0.002,
                    "accepted_evidence_sets": [[{"document_id": "nvda-excerpt", "page": 1}]],
                },
                {
                    "entity": "Microsoft",
                    "metric": "operating_income",
                    "period": "FY2024",
                    "value": 109.433,
                    "unit": "billion",
                    "abs_tolerance": 0.002,
                    "accepted_evidence_sets": [[{"document_id": "msft-excerpt", "page": 1}]],
                },
            ],
            "allowed_documents": [
                {
                    "document_id": "nvda-excerpt",
                    "path": "tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf",
                },
                {
                    "document_id": "msft-excerpt",
                    "path": "tests/fixtures/sec/derived/msft_fy2024_10k_long_excerpt.pdf",
                },
            ],
            "forbidden_claims": [],
        }

    def test_unannotated_task_is_undetermined_not_claimed_supported(self) -> None:
        task = {
            "id": "unit-no-evidence",
            "family": "financial_fact",
            "expected_action": "answer",
            "required_facts": [{
                "entity": "NVIDIA",
                "metric": "operating_income",
                "period": "FY2025",
                "value": 81.453,
                "unit": "billion",
                "abs_tolerance": 0.002,
            }],
            "allowed_documents": [{"path": "excerpt.pdf", "document_id": "excerpt"}],
            "forbidden_claims": [],
        }
        scored = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [excerpt.pdf#p1].",
        )
        self.assertTrue(scored["passed"], scored["findings"])
        self.assertTrue(scored["citation_valid"])
        self.assertEqual(scored["citation_supported"], "undetermined")
        self.assertEqual(scored["citation_support_policy_version"], CITATION_SUPPORT_POLICY_VERSION)

    def test_one_legal_cite_does_not_support_every_fact(self) -> None:
        task = self._two_issuer_task()
        both = (
            "NVIDIA FY2025 operating income was 81.453 billion USD. "
            "Microsoft FY2024 operating income was 109.433 billion USD "
            "[nvda_fy2025_10k_excerpt.pdf#p1]."
        )
        scored = score_task(task, final_output=both)
        self.assertEqual(scored["required_fact_coverage"]["hits"], 2)
        self.assertTrue(scored["citation_valid"])
        self.assertEqual(scored["citation_supported"], "failed")
        self.assertFalse(scored["passed"])
        self.assertIn("unsupported_citation", scored["findings"])
        complete = (
            "NVIDIA FY2025 operating income was 81.453 billion USD "
            "[nvda_fy2025_10k_excerpt.pdf#p1]. "
            "Microsoft FY2024 operating income was 109.433 billion USD "
            "[msft_fy2024_10k_long_excerpt.pdf#p1]."
        )
        ok = score_task(task, final_output=complete)
        self.assertEqual(ok["citation_supported"], "passed")
        self.assertTrue(ok["passed"], ok["findings"])

    def test_same_page_number_from_another_document_is_not_support(self) -> None:
        task = {
            "id": "unit-page-mix",
            "family": "multipage_period",
            "expected_action": "answer",
            "required_facts": [{
                "entity": "NVIDIA",
                "metric": "operating_income",
                "period": "FY2024",
                "value": 32.972,
                "unit": "billion",
                "abs_tolerance": 0.002,
                "accepted_evidence_sets": [[{"document_id": "nvda-fy2024", "page": 2}]],
            }],
            "allowed_documents": [
                {
                    "document_id": "nvda-fy2024",
                    "path": "tests/fixtures/sec/derived/nvda_cover_fy2025_oi_fy2024.pdf",
                },
                {
                    "document_id": "nvda-unspecified",
                    "path": "tests/fixtures/sec/derived/nvda_cover_fy2025_oi_unspecified.pdf",
                },
            ],
            "forbidden_claims": [],
        }
        mixed = score_task(
            task,
            final_output=(
                "NVIDIA FY2024 operating income was 32.972 billion USD "
                "[nvda_cover_fy2025_oi_unspecified.pdf#p2]."
            ),
        )
        self.assertTrue(mixed["citation_valid"])
        self.assertEqual(mixed["citation_supported"], "failed")
        self.assertFalse(mixed["passed"])
        cover = score_task(
            task,
            final_output=(
                "NVIDIA FY2024 operating income was 32.972 billion USD "
                "[nvda_cover_fy2025_oi_fy2024.pdf#p1]."
            ),
        )
        self.assertTrue(cover["citation_valid"])
        self.assertEqual(cover["citation_supported"], "failed")
        ok = score_task(
            task,
            final_output=(
                "NVIDIA FY2024 operating income was 32.972 billion USD "
                "[nvda_cover_fy2025_oi_fy2024.pdf#p2]."
            ),
        )
        self.assertEqual(ok["citation_supported"], "passed")
        self.assertTrue(ok["passed"], ok["findings"])

    def test_alternate_evidence_set_is_accepted(self) -> None:
        task = {
            "id": "unit-alt-evidence",
            "family": "financial_fact",
            "expected_action": "answer",
            "required_facts": [{
                "entity": "NVIDIA",
                "metric": "operating_income",
                "period": "FY2025",
                "value": 81.453,
                "unit": "billion",
                "abs_tolerance": 0.002,
                "accepted_evidence_sets": [
                    [{"document_id": "excerpt", "page": 1}],
                    [{"document_id": "excerpt", "page": 3}],
                ],
            }],
            "allowed_documents": [{
                "document_id": "excerpt",
                "path": "tests/fixtures/sec/derived/nvda_fy2025_10k_excerpt.pdf",
            }],
            "forbidden_claims": [],
        }
        first = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [nvda_fy2025_10k_excerpt.pdf#p1].",
        )
        alt = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [nvda_fy2025_10k_excerpt.pdf#p3].",
        )
        wrong = score_task(
            task,
            final_output="NVIDIA FY2025 operating income was 81.453 billion USD [nvda_fy2025_10k_excerpt.pdf#p5].",
        )
        self.assertEqual(first["citation_supported"], "passed")
        self.assertTrue(first["passed"], first["findings"])
        self.assertEqual(alt["citation_supported"], "passed")
        self.assertTrue(alt["passed"], alt["findings"])
        self.assertTrue(wrong["citation_valid"])
        self.assertEqual(wrong["citation_supported"], "failed")
        self.assertFalse(wrong["passed"])

