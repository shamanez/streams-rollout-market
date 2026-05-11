"""Launch a local HTTP server over runs/ and open the index in a browser.

Generates `runs/live/index.html` that links every produced dashboard
(dense mismatch, MoE router, agent trajectory, marketplace simulation)
with a one-line headline pulled out of the dashboard's JSON. Then serves
`runs/` over plain http.server and opens the index page.

This is the "click here to see everything" entry point.

Usage::

    python scripts/live/serve_dashboards.py            # default: port 8765
    python scripts/live/serve_dashboards.py --port 9000
    python scripts/live/serve_dashboards.py --no-browser
"""

from __future__ import annotations

import argparse
import functools
import html as _html
import http.server
import json
import socketserver
import threading
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "runs"
LIVE_ROOT = RUNS_ROOT / "live"
INDEX_REL = "live/index.html"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _dense_summary(payload: dict) -> str:
    pairs = payload.get("engine_pairs", [])
    if not pairs:
        return f"{payload.get('num_runs', 0)} runs, no engine pairs yet"
    parts = []
    for p in pairs:
        parts.append(
            f"{p['rollout_engine']}→{p['trainer_engine']}: ESS={p['mean_ess']:.3f}, "
            f"max|Δ|={p['worst_max_abs_log_ratio']:.3f}"
        )
    return " | ".join(parts)


def _router_summary(payload: dict) -> str:
    pairs = payload.get("engine_pairs", [])
    if not pairs:
        return f"{payload.get('num_runs', 0)} runs"
    parts = []
    for p in pairs:
        parts.append(
            f"{p['rollout_engine']}→{p['trainer_engine']}: "
            f"flip={p['mean_router_flip_rate']:.3f}, "
            f"set_disagree={p['mean_token_expert_disagreement_rate']:.3f}"
        )
    return " | ".join(parts)


def _agent_summary(payload: dict) -> str:
    pairs = payload.get("engine_pairs", [])
    if not pairs:
        return f"{payload.get('num_runs', 0)} comparisons"
    parts = []
    for p in pairs:
        parts.append(
            f"{p['rollout_engine']}→{p['trainer_engine']}: "
            f"answer_match={p['final_answer_match_rate']:.2f}, "
            f"jaccard={p['mean_tool_call_jaccard']:.2f}, "
            f"first_div_step={(p['mean_first_divergence_step'] or 0):.1f}"
        )
    return " | ".join(parts)


def _marketplace_summary(payload: dict) -> str:
    counts = payload.get("livestore_by_state", {})
    return (
        f"jobs={payload.get('jobs_total', 0)}, "
        f"submissions={payload.get('submissions', 0)}, "
        f"by_state={counts}"
    )


DASHBOARDS = [
    {
        "title": "Dense mismatch dashboard",
        "blurb": "Per-(rollout_engine, trainer_engine) ESS / clipped / sequence-log-ratio over Qwen3-32B response tokens.",
        "html": "live/dense_dashboard/dense_dashboard.html",
        "json": "live/dense_dashboard/dense_dashboard.json",
        "summary_fn": _dense_summary,
    },
    {
        "title": "MoE router dashboard",
        "blurb": "Per-(rollout_engine, trainer_engine) router flip rate, token-expert set disagreement on Qwen3-30B-A3B.",
        "html": "live/router_dashboard/router_dashboard.html",
        "json": "live/router_dashboard/router_dashboard.json",
        "summary_fn": _router_summary,
    },
    {
        "title": "Agent trajectory dashboard",
        "blurb": "Multi-step tool-using agent runs. First-divergence step, tool-call Jaccard, final-answer match rate per engine pair.",
        "html": "live/agent_dashboard/agent_dashboard.html",
        "json": "live/agent_dashboard/agent_dashboard.json",
        "summary_fn": _agent_summary,
    },
    {
        "title": "Marketplace simulation",
        "blurb": "Synthetic-data simulation: 7 worker profiles (honest / noisy / stale / 4 toxic) drive the full validators + OPBC + LiveStore + audit stack.",
        "html_glob": "*/marketplace_simulation.html",
        "json_glob": "*/marketplace_simulation.json",
        "summary_fn": _marketplace_summary,
    },
]


def _resolve_card(card: dict) -> dict:
    """Resolve runtime paths for one dashboard card.

    Returns dict with keys ready, html_rel, summary, missing_reason.
    """
    if "html" in card:
        html_path = RUNS_ROOT / card["html"]
        json_path = RUNS_ROOT / card["json"]
        if not html_path.exists():
            return {"ready": False, "missing_reason": f"no {html_path.relative_to(REPO_ROOT)}"}
        payload = _read_json(json_path) if json_path.exists() else None
        summary = card["summary_fn"](payload or {}) if payload else "(no JSON sibling)"
        return {
            "ready": True,
            "html_rel": str(html_path.relative_to(RUNS_ROOT)),
            "summary": summary,
        }
    # Glob-form: marketplace simulations land under runs/<ts>/.
    matches_html = sorted(RUNS_ROOT.glob(card["html_glob"]))
    if not matches_html:
        return {"ready": False, "missing_reason": f"no glob match for {card['html_glob']}"}
    html_path = matches_html[-1]
    json_path = html_path.with_name(card["json_glob"].rsplit("/", 1)[-1])
    payload = _read_json(json_path) if json_path.exists() else None
    summary = card["summary_fn"](payload or {}) if payload else "(no JSON sibling)"
    return {
        "ready": True,
        "html_rel": str(html_path.relative_to(RUNS_ROOT)),
        "summary": summary,
    }


def render_index() -> str:
    cards_html = []
    for card in DASHBOARDS:
        resolved = _resolve_card(card)
        title = _html.escape(card["title"])
        blurb = _html.escape(card["blurb"])
        if resolved["ready"]:
            href = "../" + _html.escape(resolved["html_rel"])
            summary = _html.escape(resolved["summary"])
            cards_html.append(
                f'<article class="ready"><h2><a href="{href}">{title}</a></h2>'
                f"<p>{blurb}</p>"
                f"<p class='sum'>{summary}</p></article>"
            )
        else:
            cards_html.append(
                f'<article class="missing"><h2>{title}</h2>'
                f"<p>{blurb}</p>"
                f"<p class='miss'>not generated yet — {_html.escape(resolved['missing_reason'])}</p></article>"
            )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>streams-rollout-market — live dashboards</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:880px;margin:2rem auto;color:#111;padding:0 1rem}"
        "h1{font-size:1.4rem}"
        "article{border:1px solid #ddd;padding:1rem;margin:1rem 0;border-radius:6px}"
        "article.ready{border-color:#5b9}"
        "article.missing{border-color:#bbb;background:#fafafa;color:#666}"
        "h2{font-size:1.05rem;margin:0 0 .5rem 0}"
        "h2 a{color:#06c;text-decoration:none}"
        "h2 a:hover{text-decoration:underline}"
        "p{margin:.4rem 0}"
        "p.sum{font-family:ui-monospace,monospace;font-size:.85em;color:#444}"
        "p.miss{font-style:italic;color:#888}"
        "</style></head><body>"
        "<h1>streams-rollout-market — live dashboards</h1>"
        "<p>Real-data observatory for the rollout marketplace. Each card is a self-contained "
        "HTML view; click the title to open it. Generated under "
        f"<code>{_html.escape(str(LIVE_ROOT.relative_to(REPO_ROOT)))}</code>.</p>"
        + "".join(cards_html)
        + "</body></html>"
    )


def write_index() -> Path:
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    target = LIVE_ROOT / "index.html"
    target.write_text(render_index(), encoding="utf-8")
    return target


def serve(port: int, open_browser: bool) -> None:
    write_index()
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(RUNS_ROOT)
    )
    with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
        url = f"http://127.0.0.1:{port}/{INDEX_REL}"
        print(f"Serving {RUNS_ROOT} at {url}")
        print("Press Ctrl-C to stop.")
        if open_browser:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nshutting down")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve runs/ over http and open the unified dashboard index."
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    parser.add_argument(
        "--write-only",
        action="store_true",
        help="just regenerate index.html and exit (don't start a server)",
    )
    args = parser.parse_args()

    if args.write_only:
        target = write_index()
        print(f"wrote {target}")
        return 0
    serve(args.port, not args.no_browser)
    return 0


if __name__ == "__main__":
    main()
