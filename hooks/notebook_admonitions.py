# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Render Material admonitions written inside notebook cells."""

import html
import re

from nbconvert.exporters.templateexporter import default_filters

_markdown2html = default_filters["markdown2html"]

_MARKER = re.compile(
    r'<p>!!!\s+(?P<kind>[a-z]+)(?:\s+(?:"(?P<quote>[^"]*)"|&quot;(?P<entity>[^&]*)&quot;))?'
    r"(?P<lead>.*?)</p>(?P<blocks>(?:\s*<pre><code[^>]*>.*?</code></pre>)*)",
    re.S,
)

_BLOCK = re.compile(r"<pre><code[^>]*>(?P<code>.*?)</code></pre>", re.S)


def _admonition(match: re.Match) -> str:
    """Turn one rendered admonition into the HTML Material styles.

    A notebook cell reaches the page through nbconvert, so the 'admonition' extension of
    the site never sees these blocks: the marker becomes a paragraph and every indented
    paragraph behind it a code block. What nbconvert already rendered is reused - the
    marker's own paragraph keeps its inline HTML and the indented paragraphs are handed
    back to nbconvert's markdown renderer - so the result looks like the rest of the cell.
    """
    title = match.group("quote") or match.group("entity")
    head = title if title else match.group("kind").capitalize()
    parts = [
        f'<div class="admonition {match.group("kind")}">',
        f'<p class="admonition-title">{head}</p>',
    ]
    if match.group("lead").strip():
        parts.append(f'<p>{match.group("lead").strip()}</p>')
    for block in _BLOCK.finditer(match.group("blocks")):
        parts.append(_markdown2html(html.unescape(block.group("code"))))
    parts.append("</div>")
    return "".join(parts)


def on_page_content(html_content: str, page, config, files) -> str:
    """Expand the admonitions of a notebook page after it has been rendered.

    The page content of a notebook is produced by mkdocs-jupyter, which replaces the page
    renderer and never runs the markdown extensions of the site. The notebook output is
    recognised by the wrapper around it, so no other page is touched.
    """
    if "<!-- jupyter-wrapper -->" not in html_content or "!!!" not in html_content:
        return html_content
    return _MARKER.sub(_admonition, html_content)
