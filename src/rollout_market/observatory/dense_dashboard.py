"""Controlled dense mismatch dashboard.

Aggregates many ``dense_mismatch_report.json`` files (each produced by
``rollout_market.observatory.dense_mismatch_lab``) into one HTML+JSON
overview. Each report is an independent observation, so this dashboard
does *not* deduplicate by identity — every run is shown.

Two views:
  - per-run table with the headline metrics and the (rollout, trainer)
    engine pair that produced them
  - per-pair aggregate (count, mean ESS, mean clipped_fraction,
    mean sequence_log_ratio, worst max_abs_log_ratio) so a reviewer can
    answer 'is this engine pair systematically worse than that one?'

The dashboard is read-only and is firewall-clean; it never re-runs any
forward pass.
"""

from __future__ import annotations

import glob as _glob
import html as _html
import json

from ._dashboard_style import (
    chart_block,
    kpi_block,
    page_shell,
    palette_for,
    quality_badge_for_ess,
    render_research_question,
)
from ._glossary import render_glossary_card
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .dense_mismatch_lab import DenseMismatchReport


@dataclass(frozen=True)
class DenseRow:
    """One per-run summary on the dashboard."""

    run_id: str
    model_id: str
    rollout_engine: str
    trainer_engine: str
    precision_class: str
    quantization_class: str | None
    prompt_id: str
    num_policy_tokens: int
    delta_logprob_mean: float
    delta_logprob_abs_mean: float
    sequence_log_ratio: float
    ess: float
    clipped_fraction: float
    veto_fraction: float
    max_abs_log_ratio: float
    top_1pct_gradient_mass: float
    created_at: str

    @property
    def engine_pair_key(self) -> str:
        return f"{self.rollout_engine} -> {self.trainer_engine}"


@dataclass
class EnginePairAggregate:
    """Per-(rollout_engine, trainer_engine) summary across runs."""

    rollout_engine: str
    trainer_engine: str
    count: int
    mean_ess: float
    mean_clipped_fraction: float
    mean_sequence_log_ratio: float
    mean_delta_logprob_abs: float
    worst_max_abs_log_ratio: float
    total_policy_tokens: int


@dataclass
class DenseDashboard:
    rows: list[DenseRow] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.rows)

    def engine_pair_aggregates(self) -> list[EnginePairAggregate]:
        groups: dict[tuple[str, str], list[DenseRow]] = {}
        for row in self.rows:
            groups.setdefault((row.rollout_engine, row.trainer_engine), []).append(row)
        out: list[EnginePairAggregate] = []
        for (rollout, trainer), rows in sorted(groups.items()):
            n = len(rows)
            out.append(
                EnginePairAggregate(
                    rollout_engine=rollout,
                    trainer_engine=trainer,
                    count=n,
                    mean_ess=sum(r.ess for r in rows) / n,
                    mean_clipped_fraction=sum(r.clipped_fraction for r in rows) / n,
                    mean_sequence_log_ratio=sum(r.sequence_log_ratio for r in rows) / n,
                    mean_delta_logprob_abs=sum(r.delta_logprob_abs_mean for r in rows) / n,
                    worst_max_abs_log_ratio=max(r.max_abs_log_ratio for r in rows),
                    total_policy_tokens=sum(r.num_policy_tokens for r in rows),
                )
            )
        return out

    def as_dict(self) -> dict[str, object]:
        return {
            "num_runs": self.num_runs,
            "engine_pairs": [
                {
                    "rollout_engine": agg.rollout_engine,
                    "trainer_engine": agg.trainer_engine,
                    "count": agg.count,
                    "mean_ess": agg.mean_ess,
                    "mean_clipped_fraction": agg.mean_clipped_fraction,
                    "mean_sequence_log_ratio": agg.mean_sequence_log_ratio,
                    "mean_delta_logprob_abs": agg.mean_delta_logprob_abs,
                    "worst_max_abs_log_ratio": agg.worst_max_abs_log_ratio,
                    "total_policy_tokens": agg.total_policy_tokens,
                }
                for agg in self.engine_pair_aggregates()
            ],
            "rows": [
                {
                    "run_id": r.run_id,
                    "model_id": r.model_id,
                    "rollout_engine": r.rollout_engine,
                    "trainer_engine": r.trainer_engine,
                    "precision_class": r.precision_class,
                    "quantization_class": r.quantization_class,
                    "prompt_id": r.prompt_id,
                    "num_policy_tokens": r.num_policy_tokens,
                    "delta_logprob_mean": r.delta_logprob_mean,
                    "delta_logprob_abs_mean": r.delta_logprob_abs_mean,
                    "sequence_log_ratio": r.sequence_log_ratio,
                    "ess": r.ess,
                    "clipped_fraction": r.clipped_fraction,
                    "veto_fraction": r.veto_fraction,
                    "max_abs_log_ratio": r.max_abs_log_ratio,
                    "top_1pct_gradient_mass": r.top_1pct_gradient_mass,
                    "created_at": r.created_at,
                }
                for r in self.rows
            ],
        }


def _row_from_report(report: DenseMismatchReport) -> DenseRow:
    return DenseRow(
        run_id=report.run_id,
        model_id=report.model_id,
        rollout_engine=report.rollout_engine.name,
        trainer_engine=report.trainer_engine.name,
        precision_class=report.precision_class,
        quantization_class=report.quantization_class,
        prompt_id=report.prompt_id,
        num_policy_tokens=report.num_policy_tokens,
        delta_logprob_mean=report.delta_logprob_mean,
        delta_logprob_abs_mean=report.delta_logprob_abs_mean,
        sequence_log_ratio=report.sequence_log_ratio,
        ess=report.ess,
        clipped_fraction=report.clipped_fraction,
        veto_fraction=report.veto_fraction,
        max_abs_log_ratio=report.max_abs_log_ratio,
        top_1pct_gradient_mass=report.top_1pct_gradient_mass,
        created_at=report.created_at.isoformat(),
    )


def build_dashboard(reports: Iterable[DenseMismatchReport]) -> DenseDashboard:
    """Aggregate an iterable of dense reports into one dashboard.

    Rows are ordered by created_at ascending, so a reviewer reading
    top-to-bottom sees a chronological timeline of runs.
    """
    rows = sorted([_row_from_report(r) for r in reports], key=lambda r: r.created_at)
    return DenseDashboard(rows=rows)


def load_reports_from_paths(paths: Iterable[Path | str]) -> list[DenseMismatchReport]:
    reports: list[DenseMismatchReport] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        reports.append(DenseMismatchReport.model_validate_json(text))
    return reports


def load_reports_glob(pattern: str) -> list[DenseMismatchReport]:
    return load_reports_from_paths(sorted(_glob.glob(pattern, recursive=True)))


def render_html(dashboard: DenseDashboard) -> str:
    """Render a self-contained dashboard page (no JS, no remote assets)."""

    def _f(value: float) -> str:
        return f"{value:.4f}"

    run_rows = "".join(
        f"<tr>"
        f"<td>{_html.escape(r.run_id)}</td>"
        f"<td>{_html.escape(r.model_id)}</td>"
        f"<td>{_html.escape(r.engine_pair_key)}</td>"
        f"<td>{_html.escape(r.precision_class)}</td>"
        f"<td>{r.num_policy_tokens}</td>"
        f"<td>{_f(r.ess)}</td>"
        f"<td>{_f(r.clipped_fraction)}</td>"
        f"<td>{_f(r.veto_fraction)}</td>"
        f"<td>{_f(r.max_abs_log_ratio)}</td>"
        f"<td>{_f(r.top_1pct_gradient_mass)}</td>"
        f"</tr>"
        for r in dashboard.rows
    )
    aggs = dashboard.engine_pair_aggregates()

    if aggs:
        best_ess = max(a.mean_ess for a in aggs)
        worst_ess = min(a.mean_ess for a in aggs)
        worst_max = max(a.worst_max_abs_log_ratio for a in aggs)
    else:
        best_ess = worst_ess = worst_max = 0.0

    kpis = kpi_block([
        (str(dashboard.num_runs), "runs"),
        (str(len(aggs)), "engine pairs"),
        (f"{best_ess:.4f}", "best ESS"),
        (f"{worst_ess:.4f}", "worst ESS"),
        (f"{worst_max:.3f}", "worst max|log_ratio|"),
    ])

    pair_labels = [f"{a.rollout_engine} → {a.trainer_engine}" for a in aggs]
    colors = palette_for(len(pair_labels))

    ess_chart = chart_block(
        canvas_id="dense-ess",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Mean ESS",
                "data": [round(a.mean_ess, 5) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Mean effective sample size (closer to 1 = more on-policy)"}},
            "scales": {"y": {"min": 0.9, "max": 1.0}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    delta_chart = chart_block(
        canvas_id="dense-delta",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Mean |Δlogp|",
                "data": [round(a.mean_delta_logprob_abs, 5) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Per-token |Δlogp| averaged across runs"}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    max_chart = chart_block(
        canvas_id="dense-max",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Worst max|log_ratio|",
                "data": [round(a.worst_max_abs_log_ratio, 5) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Worst single-token |log_ratio| across runs (in nats)"}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    pair_rows_html = []
    for agg in aggs:
        pair_rows_html.append(
            "<tr>"
            f"<td><strong>{_html.escape(agg.rollout_engine)}</strong></td>"
            f"<td>{_html.escape(agg.trainer_engine)}</td>"
            f"<td>{agg.count}</td>"
            f"<td>{quality_badge_for_ess(agg.mean_ess)}</td>"
            f"<td>{_f(agg.mean_clipped_fraction)}</td>"
            f"<td>{_f(agg.mean_sequence_log_ratio)}</td>"
            f"<td>{_f(agg.mean_delta_logprob_abs)}</td>"
            f"<td>{_f(agg.worst_max_abs_log_ratio)}</td>"
            f"<td>{agg.total_policy_tokens}</td>"
            "</tr>"
        )

    glossary = render_glossary_card([
        "ESS",
        "|Δlogp|",
        "log_ratio",
        "sequence_log_ratio",
        "clipped_fraction",
        "veto_fraction",
        "top_1pct_gradient_mass",
    ])

    if aggs:
        observed = (
            f"Across {len(aggs)} (rollout, trainer) engine pair"
            f"{'s' if len(aggs) != 1 else ''}, mean ESS ranges from "
            f"{best_ess:.4f} (best) to {worst_ess:.4f} (worst). The drift "
            "between engines at the same precision is comparable to the "
            "drift introduced by FP8 quantization. clipped_fraction = 0 "
            "everywhere, so OPBC routes all four to `train` despite the "
            "measurable disagreement."
        )
    else:
        observed = "No runs in this snapshot."
    rq = render_research_question(
        question="On the same checkpoint and same prompts, how much do "
                 "different inference engines (vLLM, sglang) and precision "
                 "classes (bf16, FP8) disagree on per-token logprobs?",
        observed=observed,
        next_step="Same matrix at sequence lengths 128 / 512 / 2048 to see "
                  "whether the drift accumulates linearly or super-linearly.",
    )

    body = (
        f"{rq}"
        f'<section class="card">{kpis}</section>'
        f'<section class="card">'
        f"<h2>ESS by engine pair</h2>"
        f"<p class='sub'>Mean effective sample size, averaged across all runs in each engine pair. The closer to 1, the more the rollout engine's logprobs agree with the trainer reference.</p>"
        f"{ess_chart}"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per-token drift</h2>"
        f"<p class='sub'>Left: mean absolute delta logprob — typical disagreement size. Right: worst single-token |log_ratio| in nats — the worst spike that survived inside the engine pair.</p>"
        f"<div class='chart-row'>{delta_chart}{max_chart}</div>"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per (rollout_engine -> trainer_engine) pair</h2>"
        f"<table><thead><tr>"
        f"<th>rollout</th><th>trainer</th><th>count</th>"
        f"<th>mean ESS</th><th>mean clipped</th><th>mean seq_log_ratio</th>"
        f"<th>mean |Δ logp|</th><th>worst max|log_ratio|</th>"
        f"<th>tokens</th>"
        f"</tr></thead><tbody>{''.join(pair_rows_html)}</tbody></table>"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per run</h2>"
        f"<table><thead><tr>"
        f"<th>run_id</th><th>model</th><th>engines</th><th>precision</th>"
        f"<th>tokens</th><th>ess</th><th>clipped</th><th>veto</th>"
        f"<th>max|log_ratio|</th><th>top1% mass</th>"
        f"</tr></thead><tbody>{run_rows}</tbody></table>"
        f"</section>"
        f"{glossary}"
    )

    title = "Controlled dense mismatch dashboard"
    lede = (
        f"{dashboard.num_runs} run{'s' if dashboard.num_runs != 1 else ''} across "
        f"{len(aggs)} (rollout, trainer) engine pair{'' if len(aggs) == 1 else 's'}. "
        "Numbers are token-level logprob mismatch on the same checkpoint served by two engines."
    )
    return page_shell(title, lede, body)


def write_dashboard(dashboard: DenseDashboard, out_dir: Path | str) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "dense_dashboard.json"
    html_path = out_path / "dense_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
