"""MoE router mismatch dashboard.

Aggregates many ``router_mismatch_report.json`` files (each produced by
``rollout_market.observatory.router_mismatch_lab``) into one HTML+JSON
overview. Each run is a distinct observation — different prompts or
checkpoints — so all runs are kept and ordered by ``created_at``.

Two views, mirroring the dense dashboard:
  - per-(rollout_engine, trainer_engine) aggregate (mean flip rate,
    mean token expert disagreement, worst per-layer flip rate, total
    tokens)
  - per-run detail with the layer flip-rate summary (min/mean/max
    across layers).
"""

from __future__ import annotations

import glob as _glob
import html as _html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from ._dashboard_style import (
    chart_block,
    kpi_block,
    page_shell,
    palette_for,
)
from ._glossary import render_glossary_card

from .router_mismatch_lab import RouterMismatchReport


@dataclass(frozen=True)
class RouterRow:
    """One per-run summary on the router dashboard."""

    run_id: str
    model_id: str
    rollout_engine: str
    trainer_engine: str
    num_tokens: int
    num_layers: int
    top_k: int
    num_experts: int
    router_flip_rate: float
    token_expert_disagreement_rate: float
    layer_flip_rate_min: float
    layer_flip_rate_mean: float
    layer_flip_rate_max: float
    created_at: str

    @property
    def engine_pair_key(self) -> str:
        return f"{self.rollout_engine} -> {self.trainer_engine}"


@dataclass
class RouterEnginePairAggregate:
    """Per-(rollout_engine, trainer_engine) summary across router runs."""

    rollout_engine: str
    trainer_engine: str
    count: int
    mean_router_flip_rate: float
    mean_token_expert_disagreement_rate: float
    worst_layer_flip_rate: float
    total_tokens: int
    layer_count: int


@dataclass
class RouterDashboard:
    rows: list[RouterRow] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.rows)

    def engine_pair_aggregates(self) -> list[RouterEnginePairAggregate]:
        groups: dict[tuple[str, str], list[RouterRow]] = {}
        for row in self.rows:
            groups.setdefault((row.rollout_engine, row.trainer_engine), []).append(row)
        out: list[RouterEnginePairAggregate] = []
        for (rollout, trainer), rows in sorted(groups.items()):
            n = len(rows)
            out.append(
                RouterEnginePairAggregate(
                    rollout_engine=rollout,
                    trainer_engine=trainer,
                    count=n,
                    mean_router_flip_rate=sum(r.router_flip_rate for r in rows) / n,
                    mean_token_expert_disagreement_rate=(
                        sum(r.token_expert_disagreement_rate for r in rows) / n
                    ),
                    worst_layer_flip_rate=max(r.layer_flip_rate_max for r in rows),
                    total_tokens=sum(r.num_tokens for r in rows),
                    layer_count=max(r.num_layers for r in rows),
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
                    "mean_router_flip_rate": agg.mean_router_flip_rate,
                    "mean_token_expert_disagreement_rate": agg.mean_token_expert_disagreement_rate,
                    "worst_layer_flip_rate": agg.worst_layer_flip_rate,
                    "total_tokens": agg.total_tokens,
                    "layer_count": agg.layer_count,
                }
                for agg in self.engine_pair_aggregates()
            ],
            "rows": [
                {
                    "run_id": r.run_id,
                    "model_id": r.model_id,
                    "rollout_engine": r.rollout_engine,
                    "trainer_engine": r.trainer_engine,
                    "num_tokens": r.num_tokens,
                    "num_layers": r.num_layers,
                    "top_k": r.top_k,
                    "num_experts": r.num_experts,
                    "router_flip_rate": r.router_flip_rate,
                    "token_expert_disagreement_rate": r.token_expert_disagreement_rate,
                    "layer_flip_rate_min": r.layer_flip_rate_min,
                    "layer_flip_rate_mean": r.layer_flip_rate_mean,
                    "layer_flip_rate_max": r.layer_flip_rate_max,
                    "created_at": r.created_at,
                }
                for r in self.rows
            ],
        }


def _row_from_report(report: RouterMismatchReport) -> RouterRow:
    layer_rates = report.layer_flip_rates or [0.0]
    return RouterRow(
        run_id=report.run_id,
        model_id=report.model_id,
        rollout_engine=report.rollout_engine.name,
        trainer_engine=report.trainer_engine.name,
        num_tokens=report.num_tokens,
        num_layers=report.num_layers,
        top_k=report.top_k,
        num_experts=report.num_experts,
        router_flip_rate=report.router_flip_rate,
        token_expert_disagreement_rate=report.token_expert_disagreement_rate,
        layer_flip_rate_min=min(layer_rates),
        layer_flip_rate_mean=sum(layer_rates) / len(layer_rates),
        layer_flip_rate_max=max(layer_rates),
        created_at=report.created_at.isoformat(),
    )


def build_dashboard(reports: Iterable[RouterMismatchReport]) -> RouterDashboard:
    """Aggregate router-mismatch reports into one dashboard.

    Rows are sorted by created_at ascending so a reviewer reading top-to-
    bottom sees a chronological timeline of router probes.
    """
    rows = sorted([_row_from_report(r) for r in reports], key=lambda r: r.created_at)
    return RouterDashboard(rows=rows)


def load_reports_from_paths(paths: Iterable[Path | str]) -> list[RouterMismatchReport]:
    reports: list[RouterMismatchReport] = []
    for path in paths:
        text = Path(path).read_text(encoding="utf-8")
        reports.append(RouterMismatchReport.model_validate_json(text))
    return reports


def load_reports_glob(pattern: str) -> list[RouterMismatchReport]:
    return load_reports_from_paths(sorted(_glob.glob(pattern, recursive=True)))


def render_html(dashboard: RouterDashboard) -> str:
    """Render a self-contained dashboard page (no JS, no remote assets)."""

    def _f(value: float) -> str:
        return f"{value:.4f}"

    pair_rows = "".join(
        f"<tr>"
        f"<td>{_html.escape(agg.rollout_engine)}</td>"
        f"<td>{_html.escape(agg.trainer_engine)}</td>"
        f"<td>{agg.count}</td>"
        f"<td>{_f(agg.mean_router_flip_rate)}</td>"
        f"<td>{_f(agg.mean_token_expert_disagreement_rate)}</td>"
        f"<td>{_f(agg.worst_layer_flip_rate)}</td>"
        f"<td>{agg.layer_count}</td>"
        f"<td>{agg.total_tokens}</td>"
        f"</tr>"
        for agg in dashboard.engine_pair_aggregates()
    )
    run_rows = "".join(
        f"<tr>"
        f"<td>{_html.escape(r.run_id)}</td>"
        f"<td>{_html.escape(r.model_id)}</td>"
        f"<td>{_html.escape(r.engine_pair_key)}</td>"
        f"<td>{r.num_tokens}</td>"
        f"<td>{r.num_layers}</td>"
        f"<td>{r.top_k}</td>"
        f"<td>{r.num_experts}</td>"
        f"<td>{_f(r.router_flip_rate)}</td>"
        f"<td>{_f(r.token_expert_disagreement_rate)}</td>"
        f"<td>{_f(r.layer_flip_rate_min)}</td>"
        f"<td>{_f(r.layer_flip_rate_mean)}</td>"
        f"<td>{_f(r.layer_flip_rate_max)}</td>"
        f"</tr>"
        for r in dashboard.rows
    )
    aggs = dashboard.engine_pair_aggregates()

    if aggs:
        worst_set_disagree = max(a.mean_token_expert_disagreement_rate for a in aggs)
        worst_flip = max(a.mean_router_flip_rate for a in aggs)
    else:
        worst_set_disagree = worst_flip = 0.0

    kpis = kpi_block([
        (str(dashboard.num_runs), "runs"),
        (str(len(aggs)), "engine pairs"),
        (f"{worst_flip * 100:.1f}%", "worst top-1 flip rate"),
        (f"{worst_set_disagree * 100:.1f}%", "worst top-k set disagreement"),
    ])

    pair_labels = [f"{a.rollout_engine} → {a.trainer_engine}" for a in aggs]
    colors = palette_for(len(pair_labels))

    flip_chart = chart_block(
        canvas_id="router-flip",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Top-1 flip rate",
                "data": [round(a.mean_router_flip_rate, 4) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Mean per-(token, layer) top-1 expert flip rate"}},
            "scales": {"y": {"min": 0, "max": 1}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    set_chart = chart_block(
        canvas_id="router-set",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Token-set disagreement rate",
                "data": [round(a.mean_token_expert_disagreement_rate, 4) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Fraction of tokens whose top-k expert *set* differed on at least one layer"}},
            "scales": {"y": {"min": 0, "max": 1}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    glossary = render_glossary_card([
        "top-1 flip rate",
        "top-k set disagreement",
        "router_flip_rate",
        "token_expert_disagreement_rate",
    ])

    body = (
        f'<section class="card">{kpis}</section>'
        f'<section class="card">'
        f"<h2>Top-1 flip vs top-k set disagreement</h2>"
        f"<p class='sub'>Left: how often the dominant routed expert changes between rollout and trainer. Right: how often <em>any</em> of the top-k experts changes on at least one layer. The gap between the two is the headline finding — quantization noise barely shifts the top-1 but reorders the rest of the top-k.</p>"
        f"<div class='chart-row'>{flip_chart}{set_chart}</div>"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per (rollout_engine -> trainer_engine) pair</h2>"
        f"<table><thead><tr>"
        f"<th>rollout</th><th>trainer</th><th>count</th>"
        f"<th>mean flip rate</th><th>mean token disagreement</th>"
        f"<th>worst layer flip</th><th>layers</th><th>tokens</th>"
        f"</tr></thead><tbody>{pair_rows}</tbody></table>"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per run</h2>"
        f"<table><thead><tr>"
        f"<th>run_id</th><th>model</th><th>engines</th>"
        f"<th>tokens</th><th>layers</th><th>top_k</th><th>experts</th>"
        f"<th>flip rate</th><th>token disagreement</th>"
        f"<th>layer min</th><th>layer mean</th><th>layer max</th>"
        f"</tr></thead><tbody>{run_rows}</tbody></table>"
        f"</section>"
        f"{glossary}"
    )

    title = "MoE router mismatch dashboard"
    lede = (
        f"{dashboard.num_runs} run{'s' if dashboard.num_runs != 1 else ''} across "
        f"{len(aggs)} (rollout, trainer) engine pair{'' if len(aggs) == 1 else 's'}. "
        "Each cell measures how often the MoE router's top-k experts disagree between two checkpoints / engines."
    )
    return page_shell(title, lede, body)


def write_dashboard(dashboard: RouterDashboard, out_dir: Path | str) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "router_dashboard.json"
    html_path = out_path / "router_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
