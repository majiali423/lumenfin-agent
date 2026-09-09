"""Allowlist HTML sanitizer for untrusted Markdown/report text.

The browser copy lives in ``static/sanitize.js`` and must keep the same
allowlist and URL policy. This module is the unit-testable specification.
"""

from __future__ import annotations

import re
from html import escape, unescape
from html.parser import HTMLParser

ALLOWED_TAGS = frozenset(
    {
        "p",
        "br",
        "hr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "ul",
        "ol",
        "li",
        "blockquote",
        "pre",
        "code",
        "strong",
        "em",
        "b",
        "i",
        "del",
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "th",
        "td",
        "a",
        "span",
    }
)
VOID_TAGS = frozenset({"br", "hr"})
ALLOWED_ATTRS: dict[str, frozenset[str]] = {
    "a": frozenset({"href", "title"}),
    "th": frozenset({"colspan", "rowspan"}),
    "td": frozenset({"colspan", "rowspan"}),
    "code": frozenset({"class"}),
    "pre": frozenset({"class"}),
}
SAFE_URL = re.compile(r"^(https?:|mailto:|#|/[^/])", re.IGNORECASE)
LANGUAGE_CLASS = re.compile(r"^language-[\w-]+$")
SCRIPT_LIKE = re.compile(
    r"(?is)<(script|style|iframe|object|embed|form|link|meta|svg|math)[\s\S]*?</\1\s*>"
    r"|<(script|style|iframe|object|embed|form|link|meta|svg|math)[^>]*>"
)


def escape_text(value: object) -> str:
    return escape("" if value is None else str(value), quote=True)


def sanitize_url(url: object) -> str:
    raw = unescape(str(url or "")).strip()
    if not raw:
        return ""
    lowered = raw.lower().replace("\0", "").replace(" ", "")
    if lowered.startswith(("javascript:", "vbscript:", "data:")):
        return ""
    if SAFE_URL.match(raw):
        return raw
    return ""


def sanitize_html(html: str) -> str:
    if not html:
        return ""
    stripped = SCRIPT_LIKE.sub("", html)
    parser = _AllowlistParser()
    parser.feed(stripped)
    parser.close()
    return parser.output()


def render_markdown(md: str, parse_fn=None) -> str:
    """Parse Markdown then allowlist-sanitize. ``parse_fn`` is optional (tests/offline)."""
    if not md:
        return ""
    try:
        html = parse_fn(md) if parse_fn is not None else escape_text(md).replace("\n", "<br>")
    except Exception:
        html = f"<p>{escape_text(md).replace(chr(10), '<br>')}</p>"
    return sanitize_html(html or "")


class _AllowlistParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._stack: list[str] = []

    def output(self) -> str:
        return "".join(self._chunks)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in ALLOWED_TAGS:
            return
        rendered = _render_start(tag, attrs)
        if rendered:
            self._chunks.append(rendered)
            if tag not in VOID_TAGS:
                self._stack.append(tag)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag not in ALLOWED_TAGS:
            return
        rendered = _render_start(tag, attrs, self_closing=True)
        if rendered:
            self._chunks.append(rendered)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag not in ALLOWED_TAGS or tag in VOID_TAGS:
            return
        if tag in self._stack:
            while self._stack:
                current = self._stack.pop()
                self._chunks.append(f"</{current}>")
                if current == tag:
                    break

    def handle_data(self, data: str) -> None:
        self._chunks.append(escape(data, quote=False))

    def handle_entityref(self, name: str) -> None:
        self._chunks.append(escape(unescape(f"&{name};"), quote=False))

    def handle_charref(self, name: str) -> None:
        self._chunks.append(escape(unescape(f"&#{name};"), quote=False))

    def close(self) -> None:
        super().close()
        while self._stack:
            self._chunks.append(f"</{self._stack.pop()}>")


def _render_start(tag: str, attrs: list[tuple[str, str | None]], *, self_closing: bool = False) -> str:
    allowed = ALLOWED_ATTRS.get(tag, frozenset())
    parts = [tag]
    for name, value in attrs:
        attr = (name or "").lower()
        if attr.startswith("on") or attr in {"style", "srcdoc"}:
            continue
        if attr not in allowed:
            continue
        raw = "" if value is None else str(value)
        if attr in {"href", "src"}:
            raw = sanitize_url(raw)
            if not raw:
                continue
        if attr == "class" and not LANGUAGE_CLASS.match(raw.strip()):
            continue
        parts.append(f'{attr}="{escape(raw, quote=True)}"')
    rendered = "<" + " ".join(parts)
    if tag in VOID_TAGS or self_closing:
        return rendered + ">"
    return rendered + ">"
