"""Publish a snapshot of the live dashboards into a flat directory.

Reads the latest HTML+JSON pair for each dashboard out of runs/live/
and writes a publishable directory with a self-contained index.html
that links every card. No relative-path gymnastics — every dashboard
is a sibling file in the target directory.

The default target is `docs/` at the repo root, which GitHub Pages
serves as https://shamanez.github.io/streams-rollout-market/. Override
with --out-dir for a temporary build (e.g., serving from the spot).

Usage::

    python scripts/live/publish_dashboards.py            # writes to docs/
    python scripts/live/publish_dashboards.py --out-dir /tmp/dash-snap
"""

from __future__ import annotations

import argparse
import html as _html
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
LIVE_ROOT = RUNS_ROOT / "live"


def _render_full_glossary_html() -> str:
    """Lazy import: scripts/ isn't a package, so we add src/ to sys.path on demand."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from rollout_market.observatory._glossary import render_full_glossary_html
    return render_full_glossary_html()


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _endpoint_summary(payload: dict) -> str:
    if not payload:
        return ""
    totals = payload.get("capability_totals", {})
    n = payload.get("num_runs", 0)
    parts = []
    for cap in ("token_ids_available", "sampled_logprobs_available", "top_logprobs_available", "seed_supported"):
        v = totals.get(cap, 0)
        parts.append(f"{cap}={v}/{n}")
    return f"{n} probes — " + ", ".join(parts)


def _dense_summary(payload: dict) -> str:
    pairs = payload.get("engine_pairs", [])
    if not pairs:
        return f"{payload.get('num_runs', 0)} runs"
    parts = [
        f"{p['rollout_engine']}→{p['trainer_engine']}: "
        f"ESS={p['mean_ess']:.3f}, max|Δ|={p['worst_max_abs_log_ratio']:.3f}"
        for p in pairs
    ]
    return " | ".join(parts)


def _router_summary(payload: dict) -> str:
    pairs = payload.get("engine_pairs", [])
    if not pairs:
        return f"{payload.get('num_runs', 0)} runs"
    parts = [
        f"{p['rollout_engine']}→{p['trainer_engine']}: "
        f"flip={p['mean_router_flip_rate']:.3f}, "
        f"set_disagree={p['mean_token_expert_disagreement_rate']:.3f}"
        for p in pairs
    ]
    return " | ".join(parts)


def _agent_summary(payload: dict) -> str:
    pairs = payload.get("engine_pairs", [])
    if not pairs:
        return f"{payload.get('num_runs', 0)} comparisons"
    parts = [
        f"{p['rollout_engine']}→{p['trainer_engine']}: "
        f"answer_match={p['final_answer_match_rate']:.2f}, "
        f"jaccard={p['mean_tool_call_jaccard']:.2f}, "
        f"first_div_step={(p['mean_first_divergence_step'] or 0):.1f}"
        for p in pairs
    ]
    return " | ".join(parts)


def _market_summary(payload: dict) -> str:
    return (
        f"jobs={payload.get('jobs_total', 0)}, "
        f"submissions={payload.get('submissions', 0)}, "
        f"by_state={payload.get('livestore_by_state', {})}"
    )


CARDS = [
    {
        "slug": "endpoint",
        "title": "Endpoint identity-gap dashboard",
        "blurb": "Free-tier API probe coverage matrix: which providers expose token IDs, logprobs, top_logprobs, seed_supported.",
        "html_src": "endpoint_dashboard/endpoint_dashboard.html",
        "json_src": "endpoint_dashboard/endpoint_dashboard.json",
        "summary_fn": _endpoint_summary,
    },
    {
        "slug": "dense",
        "title": "Dense mismatch dashboard",
        "blurb": "Per-(rollout_engine, trainer_engine) ESS / max|log_ratio| over Qwen3-32B response tokens, 8 prompts × 4 engine cells.",
        "html_src": "dense_dashboard/dense_dashboard.html",
        "json_src": "dense_dashboard/dense_dashboard.json",
        "summary_fn": _dense_summary,
    },
    {
        "slug": "router",
        "title": "MoE router dashboard",
        "blurb": "Qwen3-30B-A3B router trace: per-(token, layer) top-1 flip rate and top-k set disagreement, FP8 vs bf16.",
        "html_src": "router_dashboard/router_dashboard.html",
        "json_src": "router_dashboard/router_dashboard.json",
        "summary_fn": _router_summary,
    },
    {
        "slug": "agent",
        "title": "Agent trajectory dashboard",
        "blurb": "Multi-step tool-using Qwen3-32B runs across 4 engine cells: first-divergence step, tool-call Jaccard, final-answer match rate.",
        "html_src": "agent_dashboard/agent_dashboard.html",
        "json_src": "agent_dashboard/agent_dashboard.json",
        "summary_fn": _agent_summary,
    },
    {
        "slug": "marketplace",
        "title": "Marketplace simulation",
        "blurb": "Synthetic 7-worker simulation (honest / noisy / stale / 4 toxic) driving the full validators + OPBC + LiveStore + audit stack.",
        "html_glob": "../*/marketplace_simulation.html",
        "json_glob": "../*/marketplace_simulation.json",
        "summary_fn": _market_summary,
    },
]


def _resolve_html_json(card: dict) -> tuple[Path, Path] | None:
    if "html_src" in card:
        h = LIVE_ROOT / card["html_src"]
        j = LIVE_ROOT / card["json_src"]
        if h.exists():
            return h, j if j.exists() else h.with_suffix(".json.missing")
        return None
    matches = sorted((LIVE_ROOT).glob(card["html_glob"]))
    if not matches:
        return None
    h = matches[-1]
    j = h.with_name(card["json_glob"].rsplit("/", 1)[-1])
    return h, j


_HEADLINES: dict[str, list[tuple[str, callable]]] = {
    "agent": [
        ("worst answer-match rate",
         lambda p: f"{min((x['final_answer_match_rate'] for x in p.get('engine_pairs', [])), default=0)*100:.0f}%"),
        ("worst tool-jaccard",
         lambda p: f"{min((x['mean_tool_call_jaccard'] for x in p.get('engine_pairs', [])), default=0):.2f}"),
    ],
    "router": [
        ("worst top-1 flip rate",
         lambda p: f"{max((x['mean_router_flip_rate'] for x in p.get('engine_pairs', [])), default=0)*100:.1f}%"),
        ("worst top-k set disagreement",
         lambda p: f"{max((x['mean_token_expert_disagreement_rate'] for x in p.get('engine_pairs', [])), default=0)*100:.1f}%"),
    ],
    "dense": [
        ("worst ESS",
         lambda p: f"{min((x['mean_ess'] for x in p.get('engine_pairs', [])), default=0):.4f}"),
        ("worst max|log_ratio|",
         lambda p: f"{max((x['worst_max_abs_log_ratio'] for x in p.get('engine_pairs', [])), default=0):.3f}"),
    ],
    "endpoint": [
        ("seed_supported",
         lambda p: f"{p.get('capability_totals', {}).get('seed_supported', 0)}/{p.get('num_runs', 0)}"),
        ("sampled_logprobs",
         lambda p: f"{p.get('capability_totals', {}).get('sampled_logprobs_available', 0)}/{p.get('num_runs', 0)}"),
    ],
    "marketplace": [
        ("submissions",
         lambda p: str(p.get("submissions", 0))),
        ("rejected",
         lambda p: str(p.get("livestore_by_state", {}).get("rejected", 0))),
    ],
}


def _kpi_block_for(slug: str, payload: dict) -> str:
    items = _HEADLINES.get(slug, [])
    if not items:
        return ""
    cells = []
    for label, fn in items:
        try:
            value = fn(payload or {})
        except Exception:
            value = "—"
        cells.append(
            f"<div class='kpi'><div class='v'>{_html.escape(value)}</div>"
            f"<div class='l'>{_html.escape(label)}</div></div>"
        )
    return f"<div class='kpi-row'>{''.join(cells)}</div>"


def render_index(card_data: list[dict]) -> str:
    cards_html = []
    for c in card_data:
        title = _html.escape(c["title"])
        blurb = _html.escape(c["blurb"])
        slug = c.get("slug", "")
        if c.get("ready"):
            href = _html.escape(c["filename"])
            payload = c.get("payload") or {}
            kpis = _kpi_block_for(slug, payload)
            cards_html.append(
                f'<article class="ready">'
                f'<h2><a href="{href}">{title} →</a></h2>'
                f"<p class='blurb'>{blurb}</p>"
                f"{kpis}"
                f"</article>"
            )
        else:
            cards_html.append(
                f'<article class="missing"><h2>{title}</h2>'
                f"<p class='blurb'>{blurb}</p>"
                f"<p class='miss'>not in this snapshot</p></article>"
            )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>streams-rollout-market — live dashboards</title>"
        "<style>"
        ":root{--bg:#f6f8fb;--surface:#fff;--border:#e2e8f0;--text:#0f172a;"
        "--muted:#64748b;--accent:#2563eb}"
        "*{box-sizing:border-box}"
        "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
        "background:var(--bg);color:var(--text);max-width:1100px;margin:0 auto;"
        "padding:1.5rem 1rem 4rem;line-height:1.5}"
        "header{margin-bottom:1.5rem;text-align:center}"
        "header h1{font-size:1.8rem;margin:0 0 .4rem 0;font-weight:700;letter-spacing:-.02em}"
        "header p{color:var(--muted);margin:0 auto;max-width:640px;font-size:1rem}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1rem}"
        "article{background:var(--surface);border:1px solid var(--border);"
        "border-radius:14px;padding:1.25rem 1.5rem;box-shadow:0 1px 2px rgba(15,23,42,.04);"
        "transition:transform .12s ease, box-shadow .12s ease}"
        "article.ready:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(15,23,42,.07)}"
        "article.missing{background:#fafafa;color:var(--muted);border-style:dashed}"
        "h2{font-size:1.1rem;margin:0 0 .4rem 0;font-weight:600;letter-spacing:-.01em}"
        "h2 a{color:var(--accent);text-decoration:none}"
        "h2 a:hover{text-decoration:underline}"
        "p.blurb{color:var(--muted);font-size:.93rem;margin:0 0 .9rem 0}"
        "p.miss{color:var(--muted);font-style:italic;margin-top:.5rem;font-size:.85rem}"
        ".kpi-row{display:grid;grid-template-columns:1fr 1fr;gap:.55rem;margin-top:.7rem}"
        ".kpi{background:#f1f5f9;border-radius:8px;padding:.55rem .75rem;"
        "border:1px solid var(--border)}"
        ".kpi .v{font-size:1.05rem;font-weight:700;font-variant-numeric:tabular-nums;color:var(--text)}"
        ".kpi .l{font-size:.72rem;color:var(--muted);text-transform:uppercase;"
        "letter-spacing:.04em;margin-top:.1rem}"
        "footer{margin-top:2.5rem;color:var(--muted);font-size:.82rem;text-align:center}"
        "</style></head><body>"
        "<header>"
        "<h1>streams-rollout-market — live dashboards</h1>"
        "<p>Real-data observatory for the rollout marketplace. Five lenses on how engine "
        "and precision choices change what an LLM-driven agent <em>actually does</em>.</p>"
        "<p style='margin-top:.6rem;display:flex;gap:1.2rem;justify-content:center;flex-wrap:wrap'>"
        "<a href='glossary.html' style='color:var(--accent);text-decoration:none;font-size:.92rem'>📖 What do these metrics mean?</a>"
        "<a href='future_research.md' style='color:var(--accent);text-decoration:none;font-size:.92rem'>🔬 Future research &amp; open problems</a>"
        "<a href='https://github.com/shamanez/streams-rollout-market' style='color:var(--accent);text-decoration:none;font-size:.92rem'>📂 Source code</a>"
        "</p>"
        "</header>"
        f"<div class='grid'>{''.join(cards_html)}</div>"
        "<footer>Generated from runs/live/. Each dashboard is regenerated by running the "
        "lab + dashboard CLIs in the source repo. <a href='glossary.html' "
        "style='color:var(--muted)'>Glossary</a>.</footer>"
        "</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "docs"),
        help="target directory (default: docs/ at repo root)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    card_data: list[dict] = []
    for card in CARDS:
        resolved = _resolve_html_json(card)
        if not resolved:
            card_data.append({**card, "ready": False})
            continue
        h, j = resolved
        target_html = out_dir / f"{card['slug']}_dashboard.html"
        shutil.copy2(h, target_html)
        if j.exists():
            target_json = out_dir / f"{card['slug']}_dashboard.json"
            shutil.copy2(j, target_json)
            payload = _read_json(j) or {}
        else:
            payload = {}
        card_data.append({
            **card,
            "ready": True,
            "filename": target_html.name,
            "summary": card["summary_fn"](payload),
            "payload": payload,
        })

    (out_dir / "index.html").write_text(render_index(card_data), encoding="utf-8")
    (out_dir / "glossary.html").write_text(_render_full_glossary_html(), encoding="utf-8")
    written = sum(1 for c in card_data if c.get("ready"))
    print(f"wrote {out_dir} — {written} dashboards + index.html + glossary.html")
    return 0


if __name__ == "__main__":
    main()
