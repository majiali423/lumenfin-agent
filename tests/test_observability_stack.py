from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ObservabilityStackTestCase(unittest.TestCase):
    def test_compose_overlay_exposes_three_scrape_targets(self) -> None:
        overlay = (ROOT / "docker-compose.observability.yml").read_text(encoding="utf-8")
        prometheus = (ROOT / "deploy/observability/prometheus.yml").read_text(encoding="utf-8")
        self.assertIn("prom/prometheus:v3.14.0", overlay)
        self.assertIn("grafana/grafana:13.1.0", overlay)
        self.assertIn('MAS_METRICS_PORT: "9101"', overlay)
        self.assertIn('MAS_METRICS_PORT: "9102"', overlay)
        self.assertIn('"lumenfin-api:8000"', prometheus)
        self.assertIn('"lumenfin-worker:9101"', prometheus)
        self.assertIn('"lumenfin-index-worker:9102"', prometheus)

    def test_dashboard_is_valid_and_provisioned(self) -> None:
        dashboard_path = ROOT / "deploy/observability/grafana/dashboards/lumenfin-overview.json"
        dashboard = json.loads(dashboard_path.read_text(encoding="utf-8"))
        self.assertEqual(dashboard["uid"], "lumenfin-ops")
        self.assertGreaterEqual(len(dashboard["panels"]), 8)
        expressions = "\n".join(
            target["expr"]
            for panel in dashboard["panels"]
            for target in panel.get("targets", [])
        )
        self.assertIn("lumenfin_workflow_duration_seconds_bucket", expressions)
        self.assertIn("lumenfin_queue_depth", expressions)
        self.assertIn("lumenfin_citation_validation_total", expressions)
        provisioning = (
            ROOT / "deploy/observability/grafana/provisioning/dashboards/lumenfin.yml"
        ).read_text(encoding="utf-8")
        self.assertIn("/var/lib/grafana/dashboards", provisioning)

    def test_monitoring_source_has_no_high_cardinality_label_names(self) -> None:
        source = (ROOT / "src/lumenfin/monitoring.py").read_text(encoding="utf-8")
        forbidden_label_declarations = (
            '("tenant_id",', '("session_id",', '("thread_id",', '("request_id",',
            '("document_id",', '("chunk_id",', '("query",', '("company",',
        )
        for declaration in forbidden_label_declarations:
            self.assertNotIn(declaration, source)

    def test_documentation_keeps_monitoring_and_accuracy_separate(self) -> None:
        doc = (ROOT / "docs/OBSERVABILITY.md").read_text(encoding="utf-8")
        self.assertIn("Prometheus 数据不是 FinAgentBench 分数或产品准确率", doc)
        self.assertIn("FinAgentBench仍负责离线确定性合同与CI门禁", doc)


if __name__ == "__main__":
    unittest.main()
