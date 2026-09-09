from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "static" / "index.html"
APP_JS = ROOT / "static" / "app.js"


class FrontendBenchDrawerTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = INDEX.read_text(encoding="utf-8")
        cls.js = APP_JS.read_text(encoding="utf-8")

    def test_drawer_is_hidden_until_the_interviewer_opens_it(self) -> None:
        self.assertIn('id="benchTrigger"', self.html)
        self.assertIn('aria-expanded="false"', self.html)
        self.assertIn('id="benchOverlay" aria-hidden="true"', self.html)
        self.assertIn('role="dialog"', self.html)

    def test_drawer_presents_the_sealed_finagentbench_evidence(self) -> None:
        self.assertIn("FINAGENTBENCH · OFFLINE REPLAY", self.html)
        self.assertIn("14 / 14", self.html)
        self.assertIn("11 / 11", self.html)
        self.assertIn("不代表真实流量检出率", self.html)
        self.assertIn("NVIDIA × AMD 对比研究", self.html)
        self.assertIn("干净 FinRun 通过", self.html)

    def test_drawer_does_not_mislabel_case_score_as_product_accuracy(self) -> None:
        self.assertIn("离线 Case 合同分", self.html)
        self.assertIn("不代表通用金融准确率", self.html)
        self.assertIn("不是当前页面这次分析的实时得分", self.html)
        self.assertIn("不是本次产品准确率", self.html)

    def test_optional_icon_cdn_cannot_break_core_page_javascript(self) -> None:
        self.assertIn("function refreshIcons()", self.js)
        self.assertIn("if (window.lucide", self.js)
        self.assertNotIn("\nlucide.createIcons();", self.js)
        self.assertNotIn("cdn.jsdelivr.net", self.html)


if __name__ == "__main__":
    unittest.main()
