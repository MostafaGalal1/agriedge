"""Render the manuscript to a self-contained HTML page.

Regenerate after editing manuscript.md:
    python paper/build_html.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "manuscript.md"
TARGET = HERE / "manuscript.html"

TITLE = "Provenance, Not Behaviour: an Edge-IIoTset leakage audit"

STYLE = """
:root {
  --paper:        #F6F7F5;
  --surface:      #FFFFFF;
  --ink:          #1A1F23;
  --muted:        #5F6A63;
  --rule:         #D6DAD4;
  --rule-strong:  #B4BCB4;
  --warn:         #B4551F;
  --warn-bg:      #F6EDE6;
  --ok:           #4A6B4F;
  --ok-bg:        #ECF1EC;
  --code-bg:      #EEF0EC;

  --measure: 68ch;
  --serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Source Serif 4",
           Georgia, Cambria, "Times New Roman", serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas,
          "Liberation Mono", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, sans-serif;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:       #14171A;
    --surface:     #1B1F22;
    --ink:         #E3E7E2;
    --muted:       #99A39B;
    --rule:        #2C3236;
    --rule-strong: #414A4D;
    --warn:        #E08A4F;
    --warn-bg:     #2A1E16;
    --ok:          #8FB394;
    --ok-bg:       #19231B;
    --code-bg:     #21262A;
  }
}

:root[data-theme="dark"] {
  --paper:       #14171A;
  --surface:     #1B1F22;
  --ink:         #E3E7E2;
  --muted:       #99A39B;
  --rule:        #2C3236;
  --rule-strong: #414A4D;
  --warn:        #E08A4F;
  --warn-bg:     #2A1E16;
  --ok:          #8FB394;
  --ok-bg:       #19231B;
  --code-bg:     #21262A;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--serif);
  font-size: 1.0625rem;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}

.wrap {
  max-width: 78rem;
  margin: 0 auto;
  padding: clamp(2rem, 5vw, 4.5rem) clamp(1.1rem, 4vw, 3rem) 6rem;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0;
}

/* Running text stays at a readable measure; tables may exceed it. */
.wrap > p,
.wrap > ul,
.wrap > ol,
.wrap > blockquote,
.wrap > h1,
.wrap > h2,
.wrap > h3,
.wrap > h4,
.wrap > hr { max-width: var(--measure); }

h1 {
  font-size: clamp(1.85rem, 4.4vw, 2.9rem);
  line-height: 1.14;
  font-weight: 600;
  letter-spacing: -0.017em;
  text-wrap: balance;
  margin: 0 0 1.6rem;
  max-width: 24ch !important;
}

h2 {
  font-size: clamp(1.3rem, 2.6vw, 1.62rem);
  line-height: 1.24;
  font-weight: 600;
  letter-spacing: -0.01em;
  text-wrap: balance;
  margin: 3.6rem 0 1rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--rule);
}

h3 {
  font-size: clamp(1.08rem, 2vw, 1.24rem);
  line-height: 1.3;
  font-weight: 600;
  text-wrap: balance;
  margin: 2.5rem 0 0.7rem;
}

h4 {
  font-family: var(--sans);
  font-size: 0.82rem;
  font-weight: 650;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 2rem 0 0.5rem;
}

p { margin: 0 0 1.05rem; }

strong { font-weight: 650; }

a { color: var(--warn); text-decoration-thickness: 1px; text-underline-offset: 2px; }
a:focus-visible {
  outline: 2px solid var(--warn);
  outline-offset: 3px;
  border-radius: 2px;
}

code {
  font-family: var(--mono);
  font-size: 0.855em;
  background: var(--code-bg);
  padding: 0.1em 0.36em;
  border-radius: 3px;
  border: 1px solid var(--rule);
  white-space: nowrap;
}

pre {
  background: var(--code-bg);
  border: 1px solid var(--rule);
  border-radius: 5px;
  padding: 0.95rem 1.1rem;
  overflow-x: auto;
  max-width: var(--measure);
  font-size: 0.9rem;
  line-height: 1.5;
}
pre code {
  background: none;
  border: none;
  padding: 0;
  white-space: pre;
  font-size: 1em;
}

ul, ol { margin: 0 0 1.05rem; padding-left: 1.35rem; }
li { margin-bottom: 0.42rem; }
li::marker { color: var(--muted); }

hr {
  border: 0;
  border-top: 1px solid var(--rule);
  margin: 2.8rem 0;
}

/* --- Tables ------------------------------------------------------------- */

.tablewrap {
  overflow-x: auto;
  margin: 1.3rem 0 1.9rem;
  border: 1px solid var(--rule);
  border-radius: 5px;
  background: var(--surface);
  max-width: 100%;
}

table {
  border-collapse: collapse;
  width: 100%;
  font-family: var(--sans);
  font-size: 0.855rem;
  font-variant-numeric: tabular-nums;
  line-height: 1.42;
}

thead th {
  text-align: left;
  font-weight: 620;
  font-size: 0.76rem;
  letter-spacing: 0.035em;
  text-transform: uppercase;
  color: var(--muted);
  background: var(--code-bg);
  padding: 0.6rem 0.85rem;
  border-bottom: 1px solid var(--rule-strong);
  white-space: nowrap;
}

tbody td {
  padding: 0.52rem 0.85rem;
  border-bottom: 1px solid var(--rule);
  vertical-align: top;
}
tbody tr:last-child td { border-bottom: none; }

tbody td:first-child { font-weight: 550; white-space: nowrap; }

/* Numeric columns read right-aligned once past the label column. */
tbody td:not(:first-child), thead th:not(:first-child) { text-align: right; }
tbody td:not(:first-child) { font-variant-numeric: tabular-nums; }

table code { font-size: 0.9em; background: none; border: none; padding: 0; }

/* A bolded cell in a results table is a leaked / perfect value: mark it. */
tbody td strong { color: var(--warn); font-weight: 650; }

/* --- Front matter ------------------------------------------------------- */

.abstract {
  max-width: var(--measure);
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--warn);
  border-radius: 4px;
  padding: 1.4rem 1.6rem 0.5rem;
  margin: 0 0 2rem;
  font-size: 0.985rem;
}
.abstract h4 { margin-top: 0; }

.keywords {
  max-width: var(--measure);
  font-family: var(--sans);
  font-size: 0.82rem;
  color: var(--muted);
  margin-bottom: 2.5rem;
}

.byline {
  max-width: var(--measure);
  margin: -0.9rem 0 2rem;
}
.byline p { margin: 0; }
/* The affiliation is emphasised in the source; give it its own line so the
   byline reads as a byline rather than as a run-on sentence. */
.byline em {
  display: block;
  font-size: 0.9rem;
  color: var(--muted);
  font-style: normal;
  margin-top: 0.15rem;
}

/* --- Section numbering rail (wide screens only) ------------------------- */

@media (min-width: 1080px) {
  h2 { position: relative; }
  h2[data-num]::before {
    content: attr(data-num);
    position: absolute;
    left: -4.2rem;
    top: 1.1rem;
    width: 3.2rem;
    text-align: right;
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 400;
    color: var(--muted);
    letter-spacing: 0;
  }
  .wrap { padding-left: 6rem; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""


def convert(text: str) -> str:
    """Markdown to HTML with tables and fenced code."""
    return markdown.markdown(
        text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list"],
        output_format="html5",
    )


def postprocess(html: str) -> str:
    """Wrap tables for horizontal scroll and tag h2s with their section number."""
    html = re.sub(
        r"(<table>.*?</table>)",
        r'<div class="tablewrap">\1</div>',
        html,
        flags=re.DOTALL,
    )

    def number_heading(match: re.Match[str]) -> str:
        inner = match.group(1)
        num = re.match(r"\s*(\d+)\.\s", inner)
        if not num:
            return match.group(0)
        return f'<h2 data-num="{num.group(1)}">{inner}</h2>'

    return re.sub(r"<h2>(.*?)</h2>", number_heading, html, flags=re.DOTALL)


def extract_byline(html: str) -> str:
    """Wrap the author/affiliation paragraph that follows the title."""
    return re.sub(
        r"(</h1>\s*)<p>(.*?)</p>",
        r'\1<div class="byline"><p>\2</p></div>',
        html,
        count=1,
        flags=re.DOTALL,
    )


def extract_front_matter(html: str) -> str:
    """Set the abstract and keywords apart from the running text."""
    html = html.replace(
        "<p><strong>Abstract</strong></p>",
        '<div class="abstract"><h4>Abstract</h4>',
        1,
    )
    html = re.sub(
        r"(<p><strong>Keywords:</strong>.*?</p>)",
        r'</div><div class="keywords">\1</div>',
        html,
        count=1,
        flags=re.DOTALL,
    )
    return html


def to_ascii(html: str) -> str:
    """Escape every non-ASCII character as a numeric entity.

    The manuscript uses em-dashes, plus-minus signs and multiplication signs
    throughout. Emitting them as raw UTF-8 leaves the page at the mercy of
    whatever charset the host declares; a server that omits one renders them
    as mojibake. Numeric entities are charset-independent, so the page reads
    correctly wherever it is served.
    """
    return html.encode("ascii", "xmlcharrefreplace").decode("ascii")


def main() -> int:
    if not SOURCE.is_file():
        print(f"error: {SOURCE} not found", file=sys.stderr)
        return 1

    body = extract_front_matter(
        extract_byline(postprocess(convert(SOURCE.read_text("utf-8"))))
    )
    page = to_ascii(
        f"<title>{TITLE}</title>\n"
        f"<style>{STYLE}</style>\n"
        f'<main class="wrap">\n{body}\n</main>\n'
    )
    TARGET.write_text(page, encoding="utf-8")

    non_ascii = [c for c in page if ord(c) > 127]
    if non_ascii:
        print(f"warning: {len(non_ascii)} non-ASCII chars remain", file=sys.stderr)
    print(f"wrote {TARGET} ({len(page):,} bytes, pure ASCII)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
