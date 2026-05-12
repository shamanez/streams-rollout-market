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
import re
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
        (
            "worst answer-match rate",
            lambda p: (
                f"{min((x['final_answer_match_rate'] for x in p.get('engine_pairs', [])), default=0) * 100:.0f}%"
            ),
        ),
        (
            "worst tool-jaccard",
            lambda p: (
                f"{min((x['mean_tool_call_jaccard'] for x in p.get('engine_pairs', [])), default=0):.2f}"
            ),
        ),
    ],
    # The cycle-2 MoE router_flip_rate KPIs (e.g. "7.1% worst top-1 flip
    # rate") were retired per operator direction 2026-05-12 ("Remove all
    # the numbers from cycle-2 MoE, I think it was wrong"). The router
    # card's kpi-row is intentionally empty until the planned native-API
    # replay path produces fresh Hermes-multi-turn routing numbers — see
    # the router-flip-rate-placeholder section above the fold and
    # docs/future_research.md for the wiring.
    "router": [],
    "dense": [
        (
            "worst ESS",
            lambda p: f"{min((x['mean_ess'] for x in p.get('engine_pairs', [])), default=0):.4f}",
        ),
        (
            "worst max|log_ratio|",
            lambda p: (
                f"{max((x['worst_max_abs_log_ratio'] for x in p.get('engine_pairs', [])), default=0):.3f}"
            ),
        ),
    ],
    "marketplace": [
        ("submissions", lambda p: str(p.get("submissions", 0))),
        ("rejected", lambda p: str(p.get("livestore_by_state", {}).get("rejected", 0))),
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


TRAINER_REFS = ["fsdp", "megatron"]
MEGATRON_PLACEHOLDER = "TBD — pending HF→Megatron conversion"


def _classify_trainer(trainer_engine: str) -> str | None:
    n = (trainer_engine or "").lower()
    if "fsdp" in n:
        return "fsdp"
    if "megatron" in n:
        return "megatron"
    return None


def _is_vllm_rollout(rollout_engine: str) -> bool:
    """True iff the rollout side is a vLLM variant (the only inference engine)."""
    return "vllm" in (rollout_engine or "").lower()


def _short_device(bucket: str) -> str:
    """Drop the parenthesised host label from a device bucket string.

    "L40S (g6e.12xlarge)" → "L40S". Keeps the column header compact in
    the tile grid; the full bucket is still in the tooltip / appendix.
    """
    if not bucket:
        return "L40S"
    if " (" in bucket:
        return bucket.split(" (", 1)[0]
    return bucket


def _matrix_cells_from_rows(
    rows: list[dict],
    metric_key: str,
    *,
    model_substring: str | None = None,
) -> dict[tuple[str, str], dict]:
    """Group rows by (trainer_ref_label, precision_x_device) for the matrix.

    Returns map: (trainer_label, column_label) -> {value, count, samples}.
    The column label is `"<precision> / <device>"` (short device name).

    Per the inference-only canonical architecture, only `vllm` rollouts
    paired with `fsdp` or `megatron` trainers populate the matrix.
    Other rollouts (sglang) and other trainers (HF) are skipped.

    `model_substring` (e.g. "Qwen3-32B" for the dense matrix, "Qwen3-30B-A3B"
    for the MoE matrix) filters rows whose `model_id` does not contain the
    substring. The dense and MoE matrices are model-bound per STEER.md, so
    rows from the other model must not contaminate this matrix's tiles.
    """
    cells: dict[tuple[str, str], dict] = {}
    for row in rows:
        trainer = _classify_trainer(row.get("trainer_engine", ""))
        if trainer is None:
            continue
        if not _is_vllm_rollout(row.get("rollout_engine", "")):
            continue
        if model_substring is not None:
            if model_substring not in (row.get("model_id") or ""):
                continue
        precision = row.get("precision_class") or row.get("precision", "—")
        device = _short_device(row.get("device_bucket") or "")
        col = f"{precision} / {device}"
        key = (trainer, col)
        bucket = cells.setdefault(key, {"sum": 0.0, "count": 0, "samples": []})
        value = row.get(metric_key)
        if value is None:
            continue
        bucket["sum"] += float(value)
        bucket["count"] += 1
        bucket["samples"].append(float(value))
    for v in cells.values():
        v["value"] = v["sum"] / v["count"] if v["count"] else None
    return cells


def _matrix_tile(
    *,
    trainer: str,
    column: str,
    cell: dict | None,
    kind: str,
    href: str,
) -> str:
    """Render a single tile in the (rows=trainer, cols=precision×device) grid.

    `kind` is "dense" (ESS, higher-is-better) or "moe" (router_flip_rate,
    lower-is-better).
    """
    if cell is None or cell.get("count", 0) == 0:
        if trainer == "megatron":
            inner = f"<div class='mx-placeholder'>{_html.escape(MEGATRON_PLACEHOLDER)}</div>"
            tile_class = "mx-tile mx-tbd"
        else:
            inner = "<div class='mx-empty'>no data</div>"
            tile_class = "mx-tile mx-empty-tile"
        return (
            f'<a class="{tile_class}" href="{_html.escape(href)}" '
            f'data-trainer="{_html.escape(trainer)}" data-column="{_html.escape(column)}">'
            f"<div class='mx-col'>{_html.escape(column)}</div>"
            f"{inner}"
            f"</a>"
        )
    value = cell["value"]
    if kind == "dense":
        good = value >= 0.99
        warn = value >= 0.95
        kind_class = "good" if good else ("warn" if warn else "bad")
        display = f"{value:.4f}"
        sub = "mean ESS"
    else:
        good = value <= 0.05
        warn = value <= 0.15
        kind_class = "good" if good else ("warn" if warn else "bad")
        display = f"{value * 100:.1f}%"
        sub = "mean top-1 flip"
    spark_max = max(cell["samples"]) if cell["samples"] else value
    spark_min = min(cell["samples"]) if cell["samples"] else value
    bar_pct = (
        0 if value is None else min(max(value if kind == "dense" else (1 - value), 0), 1) * 100
    )
    return (
        f'<a class="mx-tile mx-{kind_class}" href="{_html.escape(href)}" '
        f'data-chart="matrix-tile" data-trainer="{_html.escape(trainer)}" '
        f'data-column="{_html.escape(column)}">'
        f"<div class='mx-col'>{_html.escape(column)}</div>"
        f"<div class='mx-value'>{_html.escape(display)}</div>"
        f"<div class='mx-bar'><div class='mx-bar-fill' style='width:{bar_pct:.1f}%'></div></div>"
        f"<div class='mx-sub'>{_html.escape(sub)} · n={cell['count']}"
        f"{f' · range {spark_min:.4f}-{spark_max:.4f}' if kind == 'dense' else ''}"
        f"</div>"
        f"</a>"
    )


def _render_matrix(
    *,
    title: str,
    model: str,
    payload: dict,
    metric_key: str,
    kind: str,
    detail_href: str,
    section_slug: str,
    model_substring: str | None = None,
    blurb_html: str | None = None,
) -> str:
    rows = payload.get("rows") or []
    if model_substring is not None:
        eligible_rows = [r for r in rows if model_substring in (r.get("model_id") or "")]
    else:
        eligible_rows = rows
    cells = _matrix_cells_from_rows(rows, metric_key, model_substring=model_substring)
    # Always render a stable column for each known rollout precision so
    # empty cells (e.g. vLLM-fp8 not run yet) show as visible placeholders
    # rather than disappearing.
    known_devices = sorted(
        {_short_device(row.get("device_bucket") or "") for row in eligible_rows}
    ) or ["L40S"]
    columns: list[str] = []
    for precision in ("bf16", "fp8"):
        for device in known_devices:
            columns.append(f"{precision} / {device}")
    for _, col in cells.keys():
        if col not in columns:
            columns.append(col)
    grid_html = []
    grid_html.append("<div class='mx-grid'>")
    grid_html.append("<div class='mx-grid-head'>")
    grid_html.append("<div class='mx-trainer-head'>trainer-ref</div>")
    for col in columns:
        grid_html.append(f"<div class='mx-col-head'>{_html.escape(col)}</div>")
    grid_html.append("</div>")
    for trainer in TRAINER_REFS:
        grid_html.append("<div class='mx-row'>")
        grid_html.append(f"<div class='mx-trainer'>{_html.escape(trainer.upper())}</div>")
        for col in columns:
            cell = cells.get((trainer, col))
            grid_html.append(
                _matrix_tile(
                    trainer=trainer,
                    column=col,
                    cell=cell,
                    kind=kind,
                    href=detail_href,
                )
            )
        grid_html.append("</div>")
    grid_html.append("</div>")
    default_blurb = (
        f"Model: <code>{_html.escape(model)}</code> · rows are trainer-side "
        f"references (FSDP / Megatron), columns are <code>precision · device</code>. "
        f"Click any tile to drill into the detail dashboard."
    )
    return (
        f'<section class="mx-section" data-section="{section_slug}">'
        f"<h2>{_html.escape(title)}</h2>"
        f"<div class='mx-blurb'>{blurb_html or default_blurb}</div>"
        f"{''.join(grid_html)}"
        f"</section>"
    )


def _hermes_pair_tile(
    *,
    label: str,
    pair: dict | None,
    href: str,
    column_label: str,
    subtitle: str,
) -> str:
    """Render a single Hermes-agent traffic-light tile.

    The colour gate (locked in STEER.md):
      green if final_answer_match_rate >= 0.80
      amber if 0.50 <= final_answer_match_rate < 0.80
      red   otherwise
    Bar width = mean_tool_call_jaccard.
    """
    if pair is None:
        return (
            f'<a class="mx-tile mx-empty-tile" href="{_html.escape(href)}" '
            f'data-trainer="{_html.escape(label)}" data-column="{_html.escape(column_label)}">'
            f"<div class='mx-col'>{_html.escape(column_label)}</div>"
            f"<div class='mx-empty'>no data</div>"
            f"</a>"
        )
    # Soft match (numeric-equivalence / substring) is the load-bearing
    # signal: strict text-equality is too strict for multi-turn agent
    # answers ("...per 24-hour day..." vs "...in a 24-hour day...",
    # same number). Falls back to strict for older reports that pre-date
    # the field.
    fam_soft = float(pair.get("final_answer_match_soft_rate") or 0.0)
    fam_strict = float(pair.get("final_answer_match_rate") or 0.0)
    # Use the soft rate when present; otherwise the strict rate keeps
    # backward compatibility.
    fam = fam_soft if fam_soft > 0 else fam_strict
    jaccard = float(pair.get("mean_tool_call_jaccard") or 0.0)
    first_div = pair.get("mean_first_divergence_step")
    n = int(pair.get("count") or 0)
    # "answered" = trajectories whose final_answer soft-matched the trainer.
    answered = int(round(fam * n))
    if fam >= 0.80:
        kind_class = "good"
    elif fam >= 0.50:
        kind_class = "warn"
    else:
        kind_class = "bad"
    # Display the answered count "X/N" rather than a "0%" / "0.0" string so
    # the value is recognisable as a real measurement (not a placeholder)
    # even when strict text matching produces a zero.
    display = f"{answered}/{n}" if n else "—/—"
    bar_pct = min(max(jaccard, 0.0), 1.0) * 100
    div_str = (
        f"first-div mean step {first_div:.1f}"
        if isinstance(first_div, (int, float)) and first_div is not None
        else "first-div n/a"
    )
    sub = f"n={n} tasks · {div_str} · {answered}-of-{n} answered"
    return (
        f'<a class="mx-tile mx-{kind_class}" href="{_html.escape(href)}" '
        f'data-chart="hermes-tile" data-trainer="{_html.escape(label)}" '
        f'data-column="{_html.escape(column_label)}">'
        f"<div class='mx-col'>{_html.escape(column_label)}</div>"
        f"<div class='mx-value'>{_html.escape(display)}</div>"
        f"<div class='mx-bar'><div class='mx-bar-fill' style='width:{bar_pct:.1f}%'></div></div>"
        f"<div class='mx-sub'>{_html.escape(sub)}</div>"
        f"</a>"
    )


def _select_hermes_pairs(
    payload: dict,
    *,
    model_substring: str,
    bf16_engine: str,
    fp8_engine: str,
    seedb_engine: str,
    precision_trainer: str | None = None,
    noise_trainer: str | None = None,
) -> tuple[dict | None, dict | None]:
    """Pick the (precision-effect, noise-floor) engine_pairs from an agent_dashboard payload.

    precision-effect: rollout=fp8_engine,   trainer=precision_trainer (default bf16_engine)
    noise-floor:      rollout=seedb_engine, trainer=noise_trainer     (default bf16_engine)

    The optional ``precision_trainer`` / ``noise_trainer`` overrides let
    the dashboard hold TP constant for the precision-effect pair (e.g.
    fp8@TP=2 vs bf16@TP=2) while keeping the noise-floor pair on a
    different baseline (e.g. seedB@TP=4 vs bf16@TP=4).
    """
    precision_trainer = precision_trainer or bf16_engine
    noise_trainer = noise_trainer or bf16_engine
    pairs = [
        p
        for p in payload.get("engine_pairs", [])
        if model_substring.lower() in (p.get("rollout_engine") or "").lower()
        or model_substring.lower() in (p.get("trainer_engine") or "").lower()
    ]
    precision = next(
        (
            p
            for p in pairs
            if (p.get("rollout_engine") or "") == fp8_engine
            and (p.get("trainer_engine") or "") == precision_trainer
        ),
        None,
    )
    noise = next(
        (
            p
            for p in pairs
            if (p.get("rollout_engine") or "") == seedb_engine
            and (p.get("trainer_engine") or "") == noise_trainer
        ),
        None,
    )
    return precision, noise


def _hermes_logprob_tile(
    *,
    label: str,
    pair: dict | None,
    href: str,
    column_label: str,
) -> str:
    """Render one Hermes-Agent logprob-shift tile (operator pivot 2026-05-12).

    Reads an aggregated dense_dashboard engine_pair (mean_ess +
    mean_delta_logprob_abs over Hermes-multi-turn trajectories) and
    renders it with the same traffic-light thresholds as the legacy
    cycle-2 Dense matrix: green ≥0.99, amber 0.95–0.99, red <0.95.

    When the pair is missing — e.g. the Megatron teacher-force hasn't
    been run yet for this (model, trainer) cell — we emit a TBD tile so
    the dashboard structure stays stable across iterations and so
    readers can see at a glance what's still on the queue.
    """
    if pair is None:
        return (
            f'<a class="mx-tile mx-tbd" href="{_html.escape(href)}" '
            f'data-chart="hermes-logprob-tile" '
            f'data-trainer="{_html.escape(label)}" '
            f'data-column="{_html.escape(column_label)}">'
            f"<div class='mx-col'>{_html.escape(column_label)}</div>"
            f"<div class='mx-value'>TBD</div>"
            f"<div class='mx-bar'><div class='mx-bar-fill' style='width:0%'></div></div>"
            f"<div class='mx-sub'>no trainer-force run yet</div>"
            f"</a>"
        )
    ess = float(pair.get("mean_ess") or 0.0)
    delta_lp = float(pair.get("mean_delta_logprob_abs") or 0.0)
    count = int(pair.get("count") or 0)
    if ess >= 0.99:
        kind_class = "good"
    elif ess >= 0.95:
        kind_class = "warn"
    else:
        kind_class = "bad"
    bar_pct = max(0.0, min(1.0, ess)) * 100
    sub = f"n={count} run(s) · mean |Δlogprob|={delta_lp:.4f}"
    return (
        f'<a class="mx-tile mx-{kind_class}" href="{_html.escape(href)}" '
        f'data-chart="hermes-logprob-tile" '
        f'data-trainer="{_html.escape(label)}" '
        f'data-column="{_html.escape(column_label)}">'
        f"<div class='mx-col'>{_html.escape(column_label)}</div>"
        f"<div class='mx-value'>{ess:.4f}</div>"
        f"<div class='mx-bar'><div class='mx-bar-fill' style='width:{bar_pct:.1f}%'></div></div>"
        f"<div class='mx-sub'>{_html.escape(sub)}</div>"
        f"</a>"
    )


def _select_hermes_logprob_cells(
    dense_payload: dict,
    *,
    rollout_engine: str,
    trainer_engines: tuple[str, ...] = ("fsdp-bf16", "megatron-bf16"),
) -> list[tuple[str, dict | None]]:
    """Pick the per-trainer engine_pair entries for a given Hermes rollout.

    Returns a list of ``(trainer_label, pair_or_None)`` tuples in the
    order given by ``trainer_engines``. ``None`` means the cell hasn't
    been paired yet — the tile renderer turns this into a TBD slot.
    """
    pairs_by_trainer: dict[str, dict] = {
        (p.get("trainer_engine") or "").lower(): p
        for p in dense_payload.get("engine_pairs", [])
        if (p.get("rollout_engine") or "") == rollout_engine
    }
    return [(t, pairs_by_trainer.get(t.lower())) for t in trainer_engines]


def _render_hermes_logprob_matrix(
    *,
    title: str,
    section_slug: str,
    dense_payload: dict,
    rollout_engine: str,
    detail_href: str,
    blurb_html: str,
    trainer_engines: tuple[str, ...] = ("fsdp-bf16", "megatron-bf16"),
) -> str:
    """Render the Hermes-Agent logprob-shift matrix per operator pivot
    2026-05-12.

    Layout: one row per rollout (single Hermes-bf16 column for now), two
    tiles for trainer ∈ {FSDP, Megatron}. Reads from
    ``docs/dense_dashboard.json``'s aggregated ``engine_pairs`` (output
    of ``rollout_market.cli.dense_dashboard``), which already aggregates
    every ``runs/live/dense/**/dense_mismatch_report.json``. So:

      Hermes rollout → vLLM proxy → minimal_agent_driver →
      AgentTrajectory →
      run_fsdp_reference.py | megatron_reference_launch.sh →
      pair_hermes_dense_reports.py →
      runs/live/dense/<pair-slug>/<task>-turn<k>/dense_mismatch_report.json →
      rollout_market.cli.dense_dashboard →
      docs/dense_dashboard.json (engine_pairs) →
      this tile.

    Tiles missing a (rollout, trainer) cell render as TBD until the
    operator runs the corresponding teacher-force pass.
    """
    cells = _select_hermes_logprob_cells(
        dense_payload, rollout_engine=rollout_engine, trainer_engines=trainer_engines
    )
    grid_html = ["<div class='mx-grid'>"]
    grid_html.append("<div class='mx-grid-head'>")
    grid_html.append("<div class='mx-trainer-head'>trainer-ref</div>")
    grid_html.append("<div class='mx-col-head'>bf16 · L40S</div>")
    grid_html.append("</div>")
    for trainer_label, pair in cells:
        # Pretty trainer name: "fsdp-bf16" → "FSDP" for the row header.
        pretty = (trainer_label.split("-")[0] or trainer_label).upper()
        grid_html.append("<div class='mx-row'>")
        grid_html.append(f"<div class='mx-trainer'>{_html.escape(pretty)}</div>")
        grid_html.append(
            _hermes_logprob_tile(
                label=pretty,
                pair=pair,
                href=detail_href,
                column_label=f"{rollout_engine}",
            )
        )
        grid_html.append("</div>")
    grid_html.append("</div>")
    return (
        f'<section class="mx-section" data-section="{section_slug}">'
        f"<h2>{_html.escape(title)}</h2>"
        f"<div class='mx-blurb'>{blurb_html}</div>"
        f"{''.join(grid_html)}"
        f"</section>"
    )


def _render_hermes_section(
    *,
    title: str,
    section_slug: str,
    payload: dict,
    model_substring: str,
    bf16_engine: str,
    fp8_engine: str,
    seedb_engine: str,
    detail_href: str,
    blurb_html: str,
    precision_trainer: str | None = None,
    noise_trainer: str | None = None,
) -> str:
    """Render the Hermes-agent matrix section (2 tiles: precision-effect + noise-floor).

    Layout mirrors the dense + MoE matrices: one row, two tiles. Tiles
    are anchored to the existing chart markup so the dashboard's CSS
    grid + traffic-light rules apply unchanged.
    """
    precision_pair, noise_pair = _select_hermes_pairs(
        payload,
        model_substring=model_substring,
        bf16_engine=bf16_engine,
        fp8_engine=fp8_engine,
        seedb_engine=seedb_engine,
        precision_trainer=precision_trainer,
        noise_trainer=noise_trainer,
    )
    columns = [
        ("fp8-vs-bf16", "precision effect", precision_pair),
        ("bf16_seedB-vs-bf16", "noise floor (bf16 vs bf16, different seed)", noise_pair),
    ]
    grid_html = ["<div class='mx-grid'>"]
    grid_html.append("<div class='mx-grid-head'>")
    grid_html.append("<div class='mx-trainer-head'>comparison</div>")
    for col, _label, _ in columns:
        grid_html.append(f"<div class='mx-col-head'>{_html.escape(col)}</div>")
    grid_html.append("</div>")
    grid_html.append("<div class='mx-row'>")
    grid_html.append("<div class='mx-trainer'>HERMES</div>")
    for col, descr, pair in columns:
        grid_html.append(
            _hermes_pair_tile(
                label=descr,
                pair=pair,
                href=detail_href,
                column_label=col,
                subtitle=descr,
            )
        )
    grid_html.append("</div>")
    grid_html.append("</div>")
    return (
        f'<section class="mx-section" data-section="{section_slug}">'
        f"<h2>{_html.escape(title)}</h2>"
        f"<div class='mx-blurb'>{blurb_html}</div>"
        f"{''.join(grid_html)}"
        f"</section>"
    )


_MATRIX_STYLES = """
.mx-section{background:var(--surface);border:1px solid var(--border);border-radius:14px;
padding:1.25rem 1.5rem;margin:1rem 0;box-shadow:0 1px 2px rgba(15,23,42,.04)}
.mx-section h2{font-size:1.1rem;margin:0 0 .35rem 0;font-weight:600}
.mx-blurb{color:var(--muted);margin:0 0 1rem 0;font-size:.9rem;line-height:1.55}
.mx-blurb p{margin:0 0 .55rem 0}
.mx-blurb p:last-child{margin-bottom:0}
.mx-blurb strong{color:var(--text);font-weight:600}
.mx-blurb code{background:#f1f5f9;padding:.05rem .35rem;border-radius:4px;font-size:.85em}
.mx-grid{display:grid;gap:.5rem}
.mx-grid-head,.mx-row{display:grid;grid-template-columns:120px repeat(auto-fit,minmax(170px,1fr));
gap:.5rem;align-items:stretch}
.mx-trainer-head,.mx-col-head{color:var(--muted);font-size:.78rem;text-transform:uppercase;
letter-spacing:.04em;padding:.25rem .35rem;font-weight:600}
.mx-trainer{display:flex;align-items:center;justify-content:flex-end;padding-right:.6rem;
font-weight:700;color:#0f172a;font-size:.95rem}
.mx-tile{display:block;text-decoration:none;color:inherit;
border:1px solid var(--border);border-radius:10px;padding:.7rem .9rem;background:var(--surface);
transition:transform .1s ease, box-shadow .1s ease}
.mx-tile:hover{transform:translateY(-2px);box-shadow:0 6px 18px rgba(15,23,42,.07)}
.mx-tile.mx-good{background:#dcfce7;border-color:#86efac}
.mx-tile.mx-warn{background:#fef3c7;border-color:#fde68a}
.mx-tile.mx-bad{background:#fee2e2;border-color:#fca5a5}
.mx-tile.mx-tbd{background:#f1f5f9;border-style:dashed;color:#475569}
.mx-tile.mx-empty-tile{background:#f8fafc;border-style:dashed;color:#94a3b8}
.mx-col{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.04em;
font-weight:600;margin-bottom:.2rem}
.mx-value{font-size:1.35rem;font-weight:700;font-variant-numeric:tabular-nums;color:#0f172a}
.mx-bar{position:relative;height:6px;background:rgba(15,23,42,.08);border-radius:3px;
margin:.4rem 0 .35rem 0;overflow:hidden}
.mx-bar-fill{height:100%;background:#2563eb;border-radius:3px}
.mx-tile.mx-good .mx-bar-fill{background:#16a34a}
.mx-tile.mx-warn .mx-bar-fill{background:#ca8a04}
.mx-tile.mx-bad .mx-bar-fill{background:#dc2626}
.mx-sub{font-size:.74rem;color:var(--muted);margin-top:.2rem}
.mx-placeholder{font-size:.85rem;color:#64748b;font-style:italic;padding:.25rem 0}
.mx-empty{font-size:.85rem;color:#94a3b8;font-style:italic}
@media (max-width:680px){.mx-grid-head,.mx-row{grid-template-columns:1fr}}
""".strip()


def render_index(card_data: list[dict]) -> str:
    by_slug = {c.get("slug"): c for c in card_data}
    dense_payload = (by_slug.get("dense") or {}).get("payload") or {}
    hermes_dense_blurb = (
        "<p><strong>What this measures.</strong> A real "
        "<a href='https://github.com/NousResearch/hermes-agent'>Hermes-Agent</a> "
        "multi-turn trajectory captured through the proxy at "
        "<code>scripts/live/logprob_capture_proxy.py</code>, then "
        "teacher-forced through FSDP and Megatron in bf16. Each tile reports "
        "the mean <strong>ESS</strong> over the agent's response tokens — "
        "how usable that rollout would be for off-policy training. Green "
        "(ESS &gt; 0.99) means rollout and trainer agree token-for-token; "
        "amber (0.95–0.99) means the trainer needs stronger off-policy "
        "correction; red (&lt; 0.95) means the rollout would be dropped. "
        "TBD tiles are trainer-force passes still on the queue.</p>"
    )
    hermes_moe_blurb = (
        "<p><strong>Same recipe, MoE rollouts.</strong> Hermes-Agent "
        "trajectory over Qwen3-30B-A3B, teacher-forced through FSDP and "
        "Megatron in bf16. The logprob-shift here measures the same thing "
        "the Dense matrix above measures — divergence in next-token "
        "predictions between rollout and trainer engines over the agent's "
        "response tokens — applied to MoE forward passes. Router-flip-rate "
        "(<em>which experts</em> each engine picked) is the placeholder "
        "section below: it's the marketplace metric for crowdsourced MoE "
        "rollouts but is gated on vLLM exposing <code>routed_experts</code> "
        "on the OpenAI HTTP endpoint, which 0.20.x does not.</p>"
    )
    # ``dense_payload`` already holds the aggregated dense_dashboard.json
    # (engine_pairs + rows). Both Hermes matrices read from the same
    # source so the aggregation invariants stay consistent.
    hermes_dense_matrix = _render_hermes_logprob_matrix(
        title="Hermes Agent — Dense (Qwen3-32B)",
        section_slug="hermes-agent-dense-matrix",
        dense_payload=dense_payload,
        rollout_engine="hermes-qwen3-32b-bf16",
        detail_href="dense_dashboard.html",
        blurb_html=hermes_dense_blurb,
    )
    hermes_moe_matrix = _render_hermes_logprob_matrix(
        title="Hermes Agent — MoE (Qwen3-30B-A3B)",
        section_slug="hermes-agent-moe-matrix",
        dense_payload=dense_payload,
        rollout_engine="hermes-qwen3-30b-a3b-bf16",
        detail_href="dense_dashboard.html",
        blurb_html=hermes_moe_blurb,
    )
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
        # Router-flip-rate placeholder styling — a visible "TBD" slot in
        # the dashboard, deliberately not hidden behind a <details>. We
        # want readers to know the metric is coming, not assume the
        # MoE-routing story is closed.
        "section[data-section='router-flip-rate-placeholder']{"
        "margin-top:2rem;background:#fffbea;border:1px dashed #c4a23a;"
        "border-radius:14px;padding:1rem 1.25rem}"
        "section[data-section='router-flip-rate-placeholder']>h2{"
        "margin:.1rem 0 .6rem 0;font-size:1.1rem;color:#7c5a14}"
        "section[data-section='router-flip-rate-placeholder'] .tbd-note{"
        "color:#5a4a14;font-size:.92rem;margin:.4rem 0;line-height:1.55}"
        + _MATRIX_STYLES + "</style></head><body>"
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
        # Cycle 3 v2: Hermes-Agent multi-turn matrices carry the headline;
        # the legacy single-prompt Dense matrix moves into a collapsible
        # appendix below the fold per STEER.md's "Hermes-only dashboard"
        # directive. The cycle-2 single-prompt MoE router_flip_rate
        # matrix has been retired entirely from the public surface and
        # replaced with the ``router-flip-rate-placeholder`` slot below.
        f"{hermes_dense_matrix}{hermes_moe_matrix}"
        # Router-flip-rate (expert routing) placeholder. Per operator
        # direction 2026-05-12 the cycle-2 single-prompt MoE
        # router_flip_rate tiles are removed from the public surface;
        # the metric returns once we drive the native-API "replay" path
        # documented in PROGRESS.md (Hermes captures token IDs via the
        # OpenAI proxy, then a side pass through vLLM's native
        # ``LLM.generate(prompt_token_ids=...)`` harvests
        # ``routed_experts`` for those exact tokens — vLLM 0.20.x's
        # OpenAI HTTP entrypoint does not expose the field). Until then
        # this section is an explicit TBD slot, not silently empty, so
        # readers know it's coming.
        '<section class="matrix-section" data-section="router-flip-rate-placeholder">'
        "<h2>Router flip rate (TBD)</h2>"
        "<p class='tbd-note'>"
        "<strong>Status:</strong> placeholder, returning soon. "
        "Top-1 expert-id disagreement between rollout and trainer per "
        "<code>(token, layer)</code> over the MoE response tokens — "
        "expressed as <strong>router_flip_rate</strong>, the marketplace "
        "metric for whether crowdsourced MoE rollouts route to the same "
        "experts the trainer would. The earlier cycle-2 single-prompt "
        "numbers (4.3%/7.1%/4.1%/6.3% across the four <code>(rollout, "
        "trainer)</code> cells) have been retired pending a "
        "multi-turn-agent re-run on the cycle-3 v2 pipeline."
        "</p>"
        "<p class='tbd-note'>"
        "<strong>Why TBD:</strong> vLLM 0.20.x's OpenAI HTTP entrypoint "
        "does not emit <code>routed_experts</code> on the response shape "
        "(verified on 0.20.1 and 0.20.2: <code>grep -rn routed_experts "
        "vllm/entrypoints/openai/</code> returns zero hits). The field "
        "is only available on the native <code>LLM.generate()</code> "
        "API path (which <code>~/run_vllm_moe_rollout.py</code> uses "
        "for cycle-2). For Hermes-Agent multi-turn captures we need a "
        "second native-API forward pass over the captured "
        "<code>(prompt_token_ids + response_token_ids)</code> to "
        "harvest the routing trace — see <code>PROGRESS.md</code> "
        "and <code>docs/future_research.md</code> for the planned wiring."
        "</p>"
        "</section>"
        f"<div class='grid'>{''.join(cards_html)}</div>"
        "<footer>Generated from runs/live/. Each dashboard is regenerated by running the "
        "lab + dashboard CLIs in the source repo. <a href='glossary.html' "
        "style='color:var(--muted)'>Glossary</a>.</footer>"
        "</body></html>"
    )


def _copy_agent_diff_click_through(out_dir: Path) -> int:
    """Copy the latest per-(pair, task) agent_divergence_report.html into
    ``out_dir/agent_diff/<pair-slug>/<task-id>.html`` so the matrix tiles
    have a reachable detail page per task.

    The pair script writes reports under
    ``runs/live/agent_diff/<pair-slug>/<UTC-ts>-<UUID>/agent_divergence_report.html``.
    With replicates there are multiple reports per (pair, task); we
    keep the most recent (by directory mtime) so the published page
    reflects the latest run. Returns the number of pages written.
    """
    src_root = LIVE_ROOT / "agent_diff"
    if not src_root.is_dir():
        return 0
    # latest_for[(pair, task)] = (mtime, html_path)
    latest_for: dict[tuple[str, str], tuple[float, Path]] = {}
    for report_json in src_root.glob("*/*/agent_divergence_report.json"):
        try:
            payload = json.loads(report_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_id = payload.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        pair_slug = report_json.parent.parent.name
        html_path = report_json.with_name("agent_divergence_report.html")
        if not html_path.exists():
            continue
        mtime = report_json.parent.stat().st_mtime
        key = (pair_slug, task_id)
        existing = latest_for.get(key)
        if existing is None or mtime > existing[0]:
            latest_for[key] = (mtime, html_path)

    written = 0
    target_root = out_dir / "agent_diff"
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    for (pair_slug, task_id), (_mtime, html) in sorted(latest_for.items()):
        dest_dir = target_root / pair_slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Sanitize task_id for filesystem: only allow [\w.-].
        safe = re.sub(r"[^\w.\-]", "_", task_id)
        shutil.copy2(html, dest_dir / f"{safe}.html")
        written += 1
    return written


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
        card_data.append(
            {
                **card,
                "ready": True,
                "filename": target_html.name,
                "summary": card["summary_fn"](payload),
                "payload": payload,
            }
        )

    (out_dir / "index.html").write_text(render_index(card_data), encoding="utf-8")
    (out_dir / "glossary.html").write_text(_render_full_glossary_html(), encoding="utf-8")
    diff_pages = _copy_agent_diff_click_through(out_dir)
    written = sum(1 for c in card_data if c.get("ready"))
    print(
        f"wrote {out_dir} — {written} dashboards + index.html + glossary.html"
        + (f" + {diff_pages} agent_diff click-through pages" if diff_pages else "")
    )
    return 0


if __name__ == "__main__":
    main()
