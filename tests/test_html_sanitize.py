"""Allowlist sanitizer spec + static frontend wiring audit."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from lumenfin.html_sanitize import (
    ALLOWED_TAGS,
    render_markdown,
    sanitize_html,
    sanitize_url,
)


ROOT = Path(__file__).resolve().parents[1]

XSS_CORPUS = (
    '<script>alert(1)</script>',
    '<img src=x onerror=alert(1)>',
    '<a href="javascript:alert(1)">click</a>',
    '<a href="data:text/html,hi">click</a>',
    '<p onclick="alert(1)">x</p>',
    '<iframe src="https://evil.example"></iframe>',
    '<svg onload=alert(1)>',
    '"><img src=x onerror=alert(1)>',
    '<a href="vbscript:alert(1)">x</a>',
)


class HtmlSanitizeTests(unittest.TestCase):
    def test_xss_corpus_cannot_keep_active_markup(self) -> None:
        for sample in XSS_CORPUS:
            with self.subTest(sample=sample):
                cleaned = sanitize_html(sample)
                lowered = cleaned.lower()
                self.assertNotIn("<script", lowered)
                self.assertNotIn("onerror", lowered)
                self.assertNotIn("onclick", lowered)
                self.assertNotIn("onload", lowered)
                self.assertNotIn("javascript:", lowered)
                self.assertNotIn("vbscript:", lowered)
                self.assertNotIn("data:", lowered)
                self.assertNotIn("<iframe", lowered)
                self.assertNotIn("<svg", lowered)
                self.assertNotIn("<img", lowered)

    def test_safe_url_policy(self) -> None:
        self.assertEqual(sanitize_url("https://example.com/a"), "https://example.com/a")
        self.assertEqual(sanitize_url("/reports/1"), "/reports/1")
        self.assertEqual(sanitize_url("#section"), "#section")
        self.assertEqual(sanitize_url("mailto:a@b.com"), "mailto:a@b.com")
        self.assertEqual(sanitize_url("javascript:alert(1)"), "")
        self.assertEqual(sanitize_url("data:text/html,hi"), "")

    def test_tables_quotes_and_https_links_survive(self) -> None:
        html = (
            "<h2>Result</h2>"
            "<blockquote>cite me</blockquote>"
            "<table><thead><tr><th>k</th></tr></thead>"
            "<tbody><tr><td>1.2%</td></tr></tbody></table>"
            '<p>See <a href="https://example.com/10-k" title="sec">10-K</a>.</p>'
        )
        cleaned = sanitize_html(html)
        self.assertIn("<table>", cleaned)
        self.assertIn("<td>1.2%</td>", cleaned)
        self.assertIn("<blockquote>cite me</blockquote>", cleaned)
        self.assertIn('href="https://example.com/10-k"', cleaned)
        self.assertIn("10-K", cleaned)

    def test_render_markdown_without_parser_escapes_then_allows_line_breaks(self) -> None:
        out = render_markdown("<script>alert(1)</script>\nnext")
        self.assertNotIn("<script", out.lower())
        self.assertIn("&lt;script&gt;", out)
        self.assertIn("<br>", out)

    def test_browser_sanitizer_allowlist_matches_python_spec(self) -> None:
        js = (ROOT / "static" / "sanitize.js").read_text(encoding="utf-8")
        block = js.split("var ALLOWED = {", 1)[1].split("};", 1)[0]
        js_tags = {match.group(1) for match in re.finditer(r"\b([a-z0-9]+):1", block)}
        self.assertEqual(js_tags, set(ALLOWED_TAGS))

    def test_index_html_routes_markdown_and_text_through_sanitizer(self) -> None:
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        frontend = html + "\n" + js
        self.assertIn('src="/static/sanitize.js"', html)
        self.assertIn("LumenFinSanitize.renderMarkdown", frontend)
        self.assertIn("LumenFinSanitize.setText", frontend)
        self.assertIn("LumenFinSanitize.escapeText", frontend)
        self.assertIn("LumenFinSanitize.setSanitizedHtml", frontend)
        self.assertNotRegex(frontend, r"innerHTML\s*=\s*markdownToHtml")
        self.assertNotRegex(frontend, r"innerHTML\s*=\s*qs\.map")
        self.assertNotRegex(frontend, r"marked\.parse\([^)]*\)\s*\|\|")


if __name__ == "__main__":
    unittest.main()
