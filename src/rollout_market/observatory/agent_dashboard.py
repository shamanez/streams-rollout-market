"""Agent-trajectory dashboard.

Aggregates many TrajectoryDivergenceReport JSON files into a single
HTML+JSON view. The headline metric is *first_divergence_step* — at what
step a rollout engine started disagreeing with the reference. Tool-choice
disagreement rate, Jaccard, and final-answer-match round out the cell.
"""

from __future__ import annotations

import glob as _glob
import html as _html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .agent_trajectory_lab import TrajectoryDivergenceReport


@dataclass(frozen=True)
class AgentRow:
    """One per-(task, rollout_engine, trainer_engine) summary."""

    run_id: str
    task_id: str
    model_id: str
    rollout_engine: str
    trainer_engine: str
    rollout_steps: int
    trainer_steps: int
    first_divergence_step: int | None
    first_divergence_kind: str | None
    tool_choice_disagreement_rate: float
    tool_call_jaccard: float
    final_answer_match: bool
    created_at: str

    @property
    def engine_pair_key(self) -> str:
        return f"{self.rollout_engine} -> {self.trainer_engine}"


@dataclass
class AgentEnginePairAggregate:
    """Per-(rollout_engine, trainer_engine) summary across tasks."""

    rollout_engine: str
    trainer_engine: str
    count: int
    mean_first_divergence_step: float | None
    fully_matched_count: int
    diverged_count: int
    mean_tool_choice_disagreement_rate: float
    mean_tool_call_jaccard: float
    final_answer_match_rate: float
    mean_rollout_steps: float
    mean_trainer_steps: float


@dataclass
class AgentDashboard:
    rows: list[AgentRow] = field(default_factory=list)

    @property
    def num_runs(self) -> int:
        return len(self.rows)

    def engine_pair_aggregates(self) -> list[AgentEnginePairAggregate]:
        groups: dict[tuple[str, str], list[AgentRow]] = {}
        for row in self.rows:
            groups.setdefault((row.rollout_engine, row.trainer_engine), []).append(row)
        out: list[AgentEnginePairAggregate] = []
        for (rollout, trainer), rows in sorted(groups.items()):
            n = len(rows)
            div_steps = [r.first_divergence_step for r in rows if r.first_divergence_step is not None]
            mean_div = (sum(div_steps) / len(div_steps)) if div_steps else None
            fully_matched = sum(1 for r in rows if r.first_divergence_step is None)
            out.append(
                AgentEnginePairAggregate(
                    rollout_engine=rollout,
                    trainer_engine=trainer,
                    count=n,
                    mean_first_divergence_step=mean_div,
                    fully_matched_count=fully_matched,
                    diverged_count=n - fully_matched,
                    mean_tool_choice_disagreement_rate=sum(
                        r.tool_choice_disagreement_rate for r in rows
                    )
                    / n,
                    mean_tool_call_jaccard=sum(r.tool_call_jaccard for r in rows) / n,
                    final_answer_match_rate=sum(
                        1 for r in rows if r.final_answer_match
                    )
                    / n,
                    mean_rollout_steps=sum(r.rollout_steps for r in rows) / n,
                    mean_trainer_steps=sum(r.trainer_steps for r in rows) / n,
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
                    "mean_first_divergence_step": agg.mean_first_divergence_step,
                    "fully_matched_count": agg.fully_matched_count,
                    "diverged_count": agg.diverged_count,
                    "mean_tool_choice_disagreement_rate": agg.mean_tool_choice_disagreement_rate,
                    "mean_tool_call_jaccard": agg.mean_tool_call_jaccard,
                    "final_answer_match_rate": agg.final_answer_match_rate,
                    "mean_rollout_steps": agg.mean_rollout_steps,
                    "mean_trainer_steps": agg.mean_trainer_steps,
                }
                for agg in self.engine_pair_aggregates()
            ],
            "rows": [
                {
                    "run_id": r.run_id,
                    "task_id": r.task_id,
                    "model_id": r.model_id,
                    "rollout_engine": r.rollout_engine,
                    "trainer_engine": r.trainer_engine,
                    "rollout_steps": r.rollout_steps,
                    "trainer_steps": r.trainer_steps,
                    "first_divergence_step": r.first_divergence_step,
                    "first_divergence_kind": r.first_divergence_kind,
                    "tool_choice_disagreement_rate": r.tool_choice_disagreement_rate,
                    "tool_call_jaccard": r.tool_call_jaccard,
                    "final_answer_match": r.final_answer_match,
                    "created_at": r.created_at,
                }
                for r in self.rows
            ],
        }


def _row_from_report(report: TrajectoryDivergenceReport) -> AgentRow:
    return AgentRow(
        run_id=report.run_id,
        task_id=report.task_id,
        model_id=report.model_id,
        rollout_engine=report.rollout_engine.name,
        trainer_engine=report.trainer_engine.name,
        rollout_steps=report.rollout_assistant_steps,
        trainer_steps=report.trainer_assistant_steps,
        first_divergence_step=report.first_divergence_step,
        first_divergence_kind=report.first_divergence_kind,
        tool_choice_disagreement_rate=report.tool_choice_disagreement_rate,
        tool_call_jaccard=report.tool_call_jaccard,
        final_answer_match=report.final_answer_match,
        created_at=report.created_at.isoformat(),
    )


def build_dashboard(reports: Iterable[TrajectoryDivergenceReport]) -> AgentDashboard:
    rows = sorted(
        [_row_from_report(r) for r in reports],
        key=lambda r: (r.engine_pair_key, r.task_id),
    )
    return AgentDashboard(rows=rows)


def load_reports_from_paths(paths: Iterable[Path | str]) -> list[TrajectoryDivergenceReport]:
    out: list[TrajectoryDivergenceReport] = []
    for p in paths:
        out.append(
            TrajectoryDivergenceReport.model_validate_json(Path(p).read_text(encoding="utf-8"))
        )
    return out


def load_reports_glob(pattern: str) -> list[TrajectoryDivergenceReport]:
    return load_reports_from_paths(sorted(_glob.glob(pattern, recursive=True)))


def render_html(dashboard: AgentDashboard) -> str:
    def _f(value: float | None) -> str:
        if value is None:
            return "—"
        return f"{value:.3f}"

    def _i(value: int | None) -> str:
        if value is None:
            return "—"
        return str(value)

    pair_rows = "".join(
        "<tr>"
        f"<td>{_html.escape(agg.rollout_engine)}</td>"
        f"<td>{_html.escape(agg.trainer_engine)}</td>"
        f"<td>{agg.count}</td>"
        f"<td>{_f(agg.mean_first_divergence_step)}</td>"
        f"<td>{agg.fully_matched_count}/{agg.count}</td>"
        f"<td>{_f(agg.mean_tool_choice_disagreement_rate)}</td>"
        f"<td>{_f(agg.mean_tool_call_jaccard)}</td>"
        f"<td>{_f(agg.final_answer_match_rate)}</td>"
        f"<td>{_f(agg.mean_rollout_steps)}</td>"
        f"<td>{_f(agg.mean_trainer_steps)}</td>"
        "</tr>"
        for agg in dashboard.engine_pair_aggregates()
    )
    run_rows = "".join(
        "<tr>"
        f"<td>{_html.escape(r.task_id)}</td>"
        f"<td>{_html.escape(r.engine_pair_key)}</td>"
        f"<td>{r.rollout_steps}</td>"
        f"<td>{r.trainer_steps}</td>"
        f"<td>{_i(r.first_divergence_step)}</td>"
        f"<td>{_html.escape(r.first_divergence_kind or '—')}</td>"
        f"<td>{_f(r.tool_choice_disagreement_rate)}</td>"
        f"<td>{_f(r.tool_call_jaccard)}</td>"
        f"<td>{'✓' if r.final_answer_match else '✗'}</td>"
        "</tr>"
        for r in dashboard.rows
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Agent trajectory dashboard</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:1180px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem;font-variant-numeric:tabular-nums}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}"
        "h1{font-size:1.2rem}h2{font-size:1rem;margin-top:1.5rem}"
        "</style></head><body>"
        f"<h1>Agent trajectory dashboard ({dashboard.num_runs} comparison"
        f"{'' if dashboard.num_runs == 1 else 's'})</h1>"
        "<h2>Per (rollout_engine -> trainer_engine) pair</h2>"
        "<table><thead><tr>"
        "<th>rollout</th><th>trainer</th><th>count</th>"
        "<th>mean first-div step</th><th>fully matched</th>"
        "<th>mean tool-disagree</th><th>mean tool-jaccard</th>"
        "<th>answer match rate</th><th>mean rollout steps</th><th>mean trainer steps</th>"
        f"</tr></thead><tbody>{pair_rows}</tbody></table>"
        "<h2>Per task × engine pair</h2>"
        "<table><thead><tr>"
        "<th>task</th><th>engines</th><th>rollout steps</th><th>trainer steps</th>"
        "<th>first-div step</th><th>div kind</th>"
        "<th>tool-disagree</th><th>tool-jaccard</th><th>answer match</th>"
        f"</tr></thead><tbody>{run_rows}</tbody></table>"
        "</body></html>"
    )


def write_dashboard(dashboard: AgentDashboard, out_dir: Path | str) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "agent_dashboard.json"
    html_path = out_path / "agent_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
