"""Shared CSS + Chart.js helpers for the observatory dashboards.

Every dashboard's `render_html` calls these so the visual design is one
place. Chart.js is loaded from a CDN — the dashboards are already
served from GitHub Pages so the network dependency is acceptable.
"""

from __future__ import annotations

import html as _html
import json
from typing import Any

CHART_JS_CDN = '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>'

# A single design language across every dashboard. Cards on a soft
# background, generous padding, monospaced numerics, conditional badges
# for good/medium/bad scores.
STYLES = """
<style>
:root{
  --bg:#f6f8fb; --surface:#fff; --border:#e2e8f0;
  --text:#0f172a; --muted:#64748b;
  --accent:#2563eb; --accent-soft:#dbeafe;
  --good:#16a34a; --good-soft:#dcfce7;
  --warn:#ca8a04; --warn-soft:#fef3c7;
  --bad:#dc2626; --bad-soft:#fee2e2;
}
*{box-sizing:border-box}
body{
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif;
  background:var(--bg);color:var(--text);
  max-width:1100px;margin:0 auto;padding:1.5rem 1rem 4rem;
  line-height:1.5;
}
header{margin-bottom:1.5rem}
header h1{font-size:1.6rem;margin:0 0 .3rem 0;font-weight:700;letter-spacing:-.01em}
header p.lede{color:var(--muted);margin:0;font-size:1rem}
section.card{
  background:var(--surface);border:1px solid var(--border);border-radius:12px;
  padding:1.25rem 1.5rem;margin:1rem 0;
  box-shadow:0 1px 2px rgba(15,23,42,.04);
}
section.card h2{font-size:1.1rem;margin:0 0 .75rem 0;font-weight:600}
section.card p.sub{color:var(--muted);margin:0 0 1rem 0;font-size:.9rem}
.chart-wrap{position:relative;height:320px;margin:1rem 0}
.chart-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:1rem}
.chart-row .chart-wrap{height:240px}
table{
  border-collapse:collapse;width:100%;
  font-variant-numeric:tabular-nums;font-size:.92rem;
}
thead th{
  background:#f1f5f9;text-align:left;
  padding:.55rem .7rem;border-bottom:1px solid var(--border);
  font-weight:600;color:#334155;
}
tbody td,tbody th{
  padding:.5rem .7rem;border-bottom:1px solid var(--border);vertical-align:top;
}
tbody tr:hover{background:#f8fafc}
.badge{
  display:inline-block;padding:.1rem .55rem;border-radius:9999px;
  font-size:.78rem;font-weight:500;
}
.badge.good{background:var(--good-soft);color:var(--good)}
.badge.warn{background:var(--warn-soft);color:var(--warn)}
.badge.bad{background:var(--bad-soft);color:var(--bad)}
.badge.muted{background:#f1f5f9;color:var(--muted)}
code{
  font-family:ui-monospace,"SF Mono",Menlo,monospace;font-size:.85em;
  background:#f1f5f9;padding:.1rem .35rem;border-radius:4px;
}
.kpi-row{
  display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:.75rem;margin:.5rem 0 1rem;
}
.kpi{
  background:#f8fafc;border:1px solid var(--border);border-radius:10px;
  padding:.75rem 1rem;
}
.kpi .v{font-size:1.4rem;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .l{font-size:.78rem;color:var(--muted);text-transform:uppercase;letter-spacing:.04em;margin-top:.15rem}
.match-yes{color:var(--good);font-weight:600}
.match-no{color:var(--bad);font-weight:600}
.row-divergent{background:#fff8f8}
footer{margin-top:2rem;color:var(--muted);font-size:.82rem;text-align:center}
</style>
""".strip()


def chart_block(canvas_id: str, chart_type: str, data: dict[str, Any], options: dict[str, Any] | None = None) -> str:
    """Return a `<div class="chart-wrap"><canvas/></div><script>` snippet.

    Embeds the chart configuration as JSON so it is self-contained and
    no external data fetch is needed.
    """
    cfg = {"type": chart_type, "data": data, "options": options or {}}
    # When user-controlled data lands inside a <script>...</script> block via
    # JSON.stringify, the raw bytes still flow into the HTML parser. Escape
    # < > & ' so the JSON payload cannot break out of the script context or
    # leak through downstream HTML interpretation.
    cfg_json = (
        json.dumps(cfg)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("'", "\\u0027")
    )
    return (
        f'<div class="chart-wrap"><canvas id="{_html.escape(canvas_id)}"></canvas></div>'
        f"<script>"
        f"(function(){{const cfg={cfg_json};"
        f'new Chart(document.getElementById("{canvas_id}"),cfg);'
        f"}})();"
        f"</script>"
    )


def page_shell(title: str, lede: str, body: str) -> str:
    """Wrap a body fragment in the standard html/head/header/footer scaffold."""
    return (
        f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{_html.escape(title)}</title>"
        f"{CHART_JS_CDN}{STYLES}"
        f"</head><body>"
        f"<header><h1>{_html.escape(title)}</h1>"
        f"<p class='lede'>{_html.escape(lede)}</p></header>"
        f"{body}"
        f"<footer>streams-rollout-market — generated from runs/live/</footer>"
        f"</body></html>"
    )


# Common Chart.js color palettes — kept short so the dashboards look unified.
PALETTE = [
    "#2563eb",  # blue
    "#16a34a",  # green
    "#ca8a04",  # amber
    "#dc2626",  # red
    "#9333ea",  # purple
    "#0891b2",  # cyan
    "#db2777",  # pink
    "#475569",  # slate
]


def palette_for(n: int) -> list[str]:
    """Pick `n` colours from the palette, cycling if necessary."""
    out = []
    for i in range(n):
        out.append(PALETTE[i % len(PALETTE)])
    return out


def kpi_block(items: list[tuple[str, str]]) -> str:
    """Render a row of KPI tiles. Each item is (value, label)."""
    cells = "".join(
        f"<div class='kpi'><div class='v'>{_html.escape(str(v))}</div>"
        f"<div class='l'>{_html.escape(label)}</div></div>"
        for v, label in items
    )
    return f"<div class='kpi-row'>{cells}</div>"


def badge(text: str, kind: str = "muted") -> str:
    return f"<span class='badge {_html.escape(kind)}'>{_html.escape(text)}</span>"


def quality_badge_for_ess(ess: float) -> str:
    if ess >= 0.99:
        return badge(f"{ess:.4f}", "good")
    if ess >= 0.95:
        return badge(f"{ess:.4f}", "warn")
    return badge(f"{ess:.4f}", "bad")


def quality_badge_for_match_rate(rate: float) -> str:
    pct = f"{rate * 100:.0f}%"
    if rate >= 0.66:
        return badge(pct, "good")
    if rate >= 0.33:
        return badge(pct, "warn")
    return badge(pct, "bad")


def render_research_question(
    question: str,
    observed: str,
    next_step: str | None = None,
) -> str:
    """One framed 'research question -> what we observed' block.

    The visitor reads this first; the dashboard's charts and tables
    underneath act as the *evidence* for the observed answer.
    """
    extra = ""
    if next_step:
        extra = (
            f"<p class='rq-next'><strong>Next:</strong> "
            f"{_html.escape(next_step)}</p>"
        )
    return (
        "<section class='rq-card'>"
        "<style>"
        ".rq-card{background:linear-gradient(180deg,#eff6ff,#fff);"
        "border:1px solid #bfdbfe;border-left:4px solid var(--accent);"
        "border-radius:12px;padding:1.1rem 1.4rem;margin:1rem 0;"
        "box-shadow:0 1px 2px rgba(15,23,42,.04)}"
        ".rq-q{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;"
        "color:var(--accent);font-weight:600;margin:0 0 .25rem 0}"
        ".rq-question{font-size:1.05rem;font-weight:600;color:var(--text);"
        "margin:0 0 .65rem 0;line-height:1.4}"
        ".rq-a{font-size:.78rem;text-transform:uppercase;letter-spacing:.05em;"
        "color:var(--good);font-weight:600;margin:.4rem 0 .25rem 0}"
        ".rq-observed{font-size:.95rem;color:var(--text);margin:0;line-height:1.5}"
        ".rq-next{font-size:.85rem;color:var(--muted);margin:.7rem 0 0 0;"
        "padding-top:.55rem;border-top:1px solid #dbeafe}"
        "</style>"
        f"<p class='rq-q'>Research question</p>"
        f"<p class='rq-question'>{_html.escape(question)}</p>"
        f"<p class='rq-a'>What we observed</p>"
        f"<p class='rq-observed'>{_html.escape(observed)}</p>"
        f"{extra}"
        "</section>"
    )
