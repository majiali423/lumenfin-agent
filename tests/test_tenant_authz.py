"""Cross-tenant authentication / authorization boundary tests."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fastapi.testclient import TestClient

from lumenfin.api.app import create_app
from lumenfin.api.auth import (
    AuthenticatedPrincipal,
    build_principal_directory,
    resolve_effective_tenant,
)
from lumenfin.llm import LocalFallbackLLMClient
from tests.support.fakes import FakeMarketDataClient
from tests.test_graph_routing import build_test_config


class AuthPrincipalUnitTests(unittest.TestCase):
    def test_legacy_key_binds_fixed_tenant(self) -> None:
        directory = build_principal_directory(
            legacy_api_key="key-a",
            legacy_client_id="client-a",
            legacy_tenant_id="tenant-a",
            principals_json=None,
        )
        self.assertEqual(directory["key-a"].tenant_id, "tenant-a")
        self.assertEqual(directory["key-a"].client_id, "client-a")

    def test_mismatch_tenant_is_forbidden(self) -> None:
        from fastapi import HTTPException

        principal = AuthenticatedPrincipal(client_id="c", tenant_id="tenant-a")
        self.assertEqual(resolve_effective_tenant(principal, None), "tenant-a")
        self.assertEqual(resolve_effective_tenant(principal, "tenant-a"), "tenant-a")
        with self.assertRaises(HTTPException) as ctx:
            resolve_effective_tenant(principal, "tenant-b")
        self.assertEqual(ctx.exception.status_code, 403)

    def test_service_job_lookup_requires_tenant_id(self) -> None:
        from lumenfin.service import LumenFinAnalysisService

        root = ROOT / "test_artifacts" / f"job-scoped-{uuid4().hex[:8]}"
        config = replace(build_test_config(root), rag_enabled=False)
        service = LumenFinAnalysisService(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        created = service.submit_job("Analyze Apple FY2024 profitability", tenant_id="tenant-a")
        with self.assertRaises(TypeError):
            service.get_job(created["job_id"])  # type: ignore[call-arg]
        with self.assertRaises(TypeError):
            service.list_jobs()  # type: ignore[call-arg]
        found = service.get_job(created["job_id"], tenant_id="tenant-a")
        self.assertIsNotNone(found)
        self.assertIsNone(service.get_job(created["job_id"], tenant_id="tenant-b"))
        self.assertEqual(
            [item["job_id"] for item in service.list_jobs(tenant_id="tenant-a")],
            [created["job_id"]],
        )
        self.assertEqual(service.list_jobs(tenant_id="tenant-b"), [])


class CrossTenantApiTests(unittest.TestCase):
    def setUp(self) -> None:
        root = ROOT / "test_artifacts" / f"tenant-authz-{uuid4().hex[:8]}"
        self.config = replace(
            build_test_config(root),
            api_key=None,
            api_key_principals_json=(
                '{"key-a":{"client_id":"client-a","tenant_id":"tenant-a"},'
                '"key-b":{"client_id":"client-b","tenant_id":"tenant-b"}}'
            ),
            rag_enabled=False,
            rag_tenant_id="tenant-a",
            api_key_tenant_id="tenant-a",
        )
        self.app = create_app(
            self.config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        self.client = TestClient(self.app)

    def test_same_tenant_job_access_succeeds(self) -> None:
        created = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "key-a"},
            json={"query": "Analyze Apple FY2024 profitability", "export_artifacts": False},
        )
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()["job_id"]
        got = self.client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "key-a"})
        self.assertEqual(got.status_code, 200, got.text)
        self.assertEqual(got.json()["job_id"], job_id)

    def test_explicit_foreign_tenant_is_forbidden(self) -> None:
        files = {"files": ("note.txt", b"hello tenant", "text/plain")}
        response = self.client.post(
            "/api/v1/documents/index",
            headers={"X-API-Key": "key-a"},
            data={"tenant_id": "tenant-b"},
            files=files,
        )
        self.assertEqual(response.status_code, 403, response.text)

    def test_cross_tenant_job_lookup_is_not_found(self) -> None:
        created = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "key-b"},
            json={"query": "Analyze Microsoft FY2024 profitability", "export_artifacts": False},
        )
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()["job_id"]
        leaked = self.client.get(f"/api/v1/jobs/{job_id}", headers={"X-API-Key": "key-a"})
        self.assertEqual(leaked.status_code, 404, leaked.text)

    def test_cross_tenant_list_jobs_hides_foreign_jobs(self) -> None:
        created = self.client.post(
            "/api/v1/jobs",
            headers={"X-API-Key": "key-b"},
            json={"query": "Analyze Tesla FY2024 profitability", "export_artifacts": False},
        )
        self.assertEqual(created.status_code, 202, created.text)
        job_id = created.json()["job_id"]
        listed = self.client.get("/api/v1/jobs", headers={"X-API-Key": "key-a"})
        self.assertEqual(listed.status_code, 200, listed.text)
        ids = {item["job_id"] for item in listed.json()}
        self.assertNotIn(job_id, ids)

    def test_cross_tenant_clarify_checkpoint_is_not_found(self) -> None:
        from lumenfin.checkpoint_store import WorkflowCheckpointRepository

        repo = WorkflowCheckpointRepository.from_database_url(
            self.config.database_url,
            db_path=self.config.db_path,
        )
        thread_id = f"thread-{uuid4().hex[:8]}"
        repo.upsert(
            thread_id=thread_id,
            query="Analyze Apple",
            state={
                "query": "Analyze Apple",
                "companies": ["Apple"],
                "workflow_status": "needs_clarification",
                "clarification_questions": ["Which fiscal year?"],
            },
            expected_revision=0,
            tenant_id="tenant-b",
        )
        response = self.client.post(
            "/api/v1/clarify",
            headers={"X-API-Key": "key-a"},
            json={
                "thread_id": thread_id,
                "clarification": {"time_range": "FY2024"},
                "export_artifacts": False,
            },
        )
        self.assertEqual(response.status_code, 404, response.text)

    def test_legacy_key_cannot_impersonate_other_tenant(self) -> None:
        config = replace(
            build_test_config(ROOT / "test_artifacts" / f"legacy-authz-{uuid4().hex[:8]}"),
            api_key="legacy-key",
            api_key_client_id="legacy-client",
            api_key_tenant_id="tenant-legacy",
            api_key_principals_json=None,
            rag_enabled=False,
            rag_tenant_id="tenant-legacy",
        )
        app = create_app(
            config,
            llm_client=LocalFallbackLLMClient(),
            market_data_client=FakeMarketDataClient(),
        )
        client = TestClient(app)
        files = {"files": ("note.txt", b"legacy", "text/plain")}
        response = client.post(
            "/api/v1/documents/index",
            headers={"X-API-Key": "legacy-key"},
            data={"tenant_id": "tenant-other"},
            files=files,
        )
        self.assertEqual(response.status_code, 403, response.text)


if __name__ == "__main__":
    unittest.main()
