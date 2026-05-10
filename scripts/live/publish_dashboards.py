"""Publish a snapshot of the live dashboards into a flat directory.

Reads the latest HTML+JSON pair for each dashboard out of runs/live/
and writes a publishable directory with a self-contained index.html
that links every card. No relative-path gymnastics — every dashboard
is a sibling file in the target directory.

Intended workflow: this dir becomes the contents of a public
GitHub Pages-backed repo so the snapshot is shareable as a stable
URL without exposing the private code repo.

Usage::

    python scripts/live/publish_dashboards.py --out-dir /tmp/dash-snap
    # then `git init`, push to the public repo, enable Pages.
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


def render_index(card_data: list[dict]) -> str:
    cards_html = []
    for c in card_data:
        title = _html.escape(c["title"])
        blurb = _html.escape(c["blurb"])
        if c.get("ready"):
            href = _html.escape(c["filename"])
            summary = _html.escape(c.get("summary", ""))
            cards_html.append(
                f'<article class="ready"><h2><a href="{href}">{title}</a></h2>'
                f"<p>{blurb}</p>"
                f"<p class='sum'>{summary}</p></article>"
            )
        else:
            cards_html.append(
                f'<article class="missing"><h2>{title}</h2>'
                f"<p>{blurb}</p>"
                f"<p class='miss'>not in this snapshot</p></article>"
            )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>streams-rollout-market — live dashboards</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:880px;margin:2rem auto;color:#111;padding:0 1rem}"
        "h1{font-size:1.4rem}"
        "header p{color:#444}"
        "article{border:1px solid #ddd;padding:1rem;margin:1rem 0;border-radius:6px}"
        "article.ready{border-color:#5b9}"
        "article.missing{border-color:#bbb;background:#fafafa;color:#666}"
        "h2{font-size:1.05rem;margin:0 0 .5rem 0}"
        "h2 a{color:#06c;text-decoration:none}"
        "h2 a:hover{text-decoration:underline}"
        "p{margin:.4rem 0}"
        "p.sum{font-family:ui-monospace,monospace;font-size:.85em;color:#444}"
        "footer{margin-top:2rem;color:#888;font-size:.85em}"
        "</style></head><body>"
        "<header>"
        "<h1>streams-rollout-market — live dashboards</h1>"
        "<p>Real-data observatory for the rollout marketplace. Each card is a self-contained "
        "static HTML snapshot. Click the title to open the full view; the JSON sibling holds "
        "the raw aggregate data.</p>"
        "</header>"
        + "".join(cards_html)
        + "<footer>Generated from runs/live/. Each dashboard is regenerated by running the "
        "lab + dashboard CLIs in the source repo.</footer>"
        "</body></html>"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, help="target directory")
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
        })

    (out_dir / "index.html").write_text(render_index(card_data), encoding="utf-8")
    written = sum(1 for c in card_data if c.get("ready"))
    print(f"wrote {out_dir} — {written} dashboards + index.html")
    return 0


if __name__ == "__main__":
    main()
