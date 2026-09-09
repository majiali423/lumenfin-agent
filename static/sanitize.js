/* LumenFin output sanitizer. Keep allowlist in sync with src/lumenfin/html_sanitize.py. */
(function (root) {
  var ALLOWED = {
    p:1, br:1, hr:1, h1:1, h2:1, h3:1, h4:1, h5:1, h6:1,
    ul:1, ol:1, li:1, blockquote:1, pre:1, code:1,
    strong:1, em:1, b:1, i:1, del:1,
    table:1, thead:1, tbody:1, tfoot:1, tr:1, th:1, td:1,
    a:1, span:1
  };
  var VOID_TAGS = { br:1, hr:1 };
  var ALLOWED_ATTRS = {
    a: { href:1, title:1 },
    th: { colspan:1, rowspan:1 },
    td: { colspan:1, rowspan:1 },
    code: { "class":1 },
    pre: { "class":1 }
  };
  var SCRIPT_LIKE = /<(script|style|iframe|object|embed|form|link|meta|svg|math)[\s\S]*?<\/\1\s*>|<(script|style|iframe|object|embed|form|link|meta|svg|math)[^>]*>/gi;

  function escapeText(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function escapeTextContent(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function sanitizeUrl(url) {
    var raw = String(url || "").replace(/&amp;/g, "&").trim();
    if (!raw) return "";
    var lowered = raw.toLowerCase().replace(/\0/g, "").replace(/\s+/g, "");
    if (lowered.indexOf("javascript:") === 0 || lowered.indexOf("vbscript:") === 0 || lowered.indexOf("data:") === 0) {
      return "";
    }
    if (/^(https?:|mailto:|#|\/[^/])/i.test(raw)) return raw;
    return "";
  }

  function attrMap(tag) {
    return ALLOWED_ATTRS[tag] || {};
  }

  function sanitizeAttributes(tag, attrs) {
    var allowed = attrMap(tag);
    var out = [];
    for (var i = 0; i < attrs.length; i++) {
      var name = String(attrs[i].name || "").toLowerCase();
      var value = attrs[i].value == null ? "" : String(attrs[i].value);
      if (name.indexOf("on") === 0 || name === "style" || name === "srcdoc") continue;
      if (!allowed[name]) continue;
      if (name === "href" || name === "src") {
        value = sanitizeUrl(value);
        if (!value) continue;
      }
      if (name === "class" && !/^language-[\w-]+$/.test(value.trim())) continue;
      out.push(name + '="' + escapeText(value) + '"');
    }
    return out;
  }

  function sanitizeWithDom(html) {
    var doc = document.implementation.createHTMLDocument("");
    var wrap = doc.createElement("div");
    wrap.innerHTML = html;
    function walk(node) {
      var pieces = [];
      for (var child = node.firstChild; child; child = child.nextSibling) {
        if (child.nodeType === 3) {
          pieces.push(escapeTextContent(child.nodeValue));
          continue;
        }
        if (child.nodeType !== 1) continue;
        var tag = child.tagName.toLowerCase();
        if (!ALLOWED[tag]) {
          pieces.push(walk(child));
          continue;
        }
        var attrs = [];
        for (var i = 0; i < child.attributes.length; i++) {
          attrs.push({ name: child.attributes[i].name, value: child.attributes[i].value });
        }
        var renderedAttrs = sanitizeAttributes(tag, attrs);
        var open = "<" + tag + (renderedAttrs.length ? " " + renderedAttrs.join(" ") : "") + ">";
        if (VOID_TAGS[tag]) {
          pieces.push(open);
          continue;
        }
        pieces.push(open + walk(child) + "</" + tag + ">");
      }
      return pieces.join("");
    }
    return walk(wrap);
  }

  function sanitizeHtml(html) {
    if (!html) return "";
    var stripped = String(html).replace(SCRIPT_LIKE, "");
    if (typeof document !== "undefined" && document.implementation && document.implementation.createHTMLDocument) {
      try { return sanitizeWithDom(stripped); } catch (err) { /* fall through */ }
    }
    return escapeText(stripped);
  }

  function renderMarkdown(md, parseFn) {
    if (!md) return "";
    var html = "";
    try {
      html = parseFn ? (parseFn(md) || "") : escapeText(md).replace(/\n/g, "<br>");
    } catch (err) {
      html = "<p>" + escapeText(md).replace(/\n/g, "<br>") + "</p>";
    }
    return sanitizeHtml(html);
  }

  function setText(el, value) {
    if (!el) return;
    el.textContent = value == null ? "" : String(value);
  }

  function setSanitizedHtml(el, html) {
    if (!el) return;
    el.innerHTML = sanitizeHtml(html);
  }

  root.LumenFinSanitize = {
    escapeText: escapeText,
    escapeTextContent: escapeTextContent,
    sanitizeUrl: sanitizeUrl,
    sanitizeHtml: sanitizeHtml,
    renderMarkdown: renderMarkdown,
    setText: setText,
    setSanitizedHtml: setSanitizedHtml
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
