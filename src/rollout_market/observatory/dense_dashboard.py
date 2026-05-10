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

    pair_rows = "".join(
        f"<tr>"
        f"<td>{_html.escape(agg.rollout_engine)}</td>"
        f"<td>{_html.escape(agg.trainer_engine)}</td>"
        f"<td>{agg.count}</td>"
        f"<td>{_f(agg.mean_ess)}</td>"
        f"<td>{_f(agg.mean_clipped_fraction)}</td>"
        f"<td>{_f(agg.mean_sequence_log_ratio)}</td>"
        f"<td>{_f(agg.mean_delta_logprob_abs)}</td>"
        f"<td>{_f(agg.worst_max_abs_log_ratio)}</td>"
        f"<td>{agg.total_policy_tokens}</td>"
        f"</tr>"
        for agg in dashboard.engine_pair_aggregates()
    )
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
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Dense mismatch dashboard</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1080px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem;font-variant-numeric:tabular-nums}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}"
        "h1{font-size:1.2rem}h2{font-size:1rem;margin-top:1.5rem}"
        "</style></head><body>"
        f"<h1>Controlled dense mismatch dashboard ({dashboard.num_runs} run"
        f"{'' if dashboard.num_runs == 1 else 's'})</h1>"
        "<h2>Per (rollout_engine -> trainer_engine) pair</h2>"
        "<table><thead><tr>"
        "<th>rollout</th><th>trainer</th><th>count</th>"
        "<th>mean ess</th><th>mean clipped</th><th>mean seq_log_ratio</th>"
        "<th>mean |delta logp|</th><th>worst max|log_ratio|</th>"
        "<th>tokens</th>"
        f"</tr></thead><tbody>{pair_rows}</tbody></table>"
        "<h2>Per run</h2>"
        "<table><thead><tr>"
        "<th>run_id</th><th>model</th><th>engines</th><th>precision</th>"
        "<th>tokens</th><th>ess</th><th>clipped</th><th>veto</th>"
        "<th>max|log_ratio|</th><th>top1% mass</th>"
        f"</tr></thead><tbody>{run_rows}</tbody></table>"
        "</body></html>"
    )


def write_dashboard(dashboard: DenseDashboard, out_dir: Path | str) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "dense_dashboard.json"
    html_path = out_path / "dense_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
