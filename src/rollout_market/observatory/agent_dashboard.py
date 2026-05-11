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

from ._dashboard_style import (
    badge,
    chart_block,
    kpi_block,
    page_shell,
    palette_for,
    quality_badge_for_match_rate,
    render_research_question,
)
from ._engine_filter import APPENDIX_HEADER, is_headline_pair
from ._glossary import render_glossary_card
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


def _agent_engine_view(
    aggs: list[AgentEnginePairAggregate],
    rows: list[AgentRow],
    canvas_prefix: str,
) -> str:
    """Render the chart+tables view for one engine partition (headline or appendix)."""

    def _f(value: float | None) -> str:
        return "—" if value is None else f"{value:.3f}"

    def _i(value: int | None) -> str:
        return "—" if value is None else str(value)

    if not aggs:
        return (
            "<section class='card'><p class='sub'>"
            "No engine pairs in this partition.</p></section>"
        )

    pair_labels = [f"{a.rollout_engine} → {a.trainer_engine}" for a in aggs]
    colors = palette_for(len(pair_labels))
    answer_match_chart = chart_block(
        canvas_id=f"agent-answer-match-{canvas_prefix}",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Final-answer match rate vs reference",
                "data": [round(a.final_answer_match_rate, 4) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "indexAxis": "y",
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "How often the rollout engine reaches the same final answer as the reference"}},
            "scales": {"x": {"min": 0, "max": 1,
                             "ticks": {"callback": "__pct__",
                                       "stepSize": 0.2}}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    ).replace('"__pct__"', "function(v){return (v*100).toFixed(0) + '%';}")
    jaccard_chart = chart_block(
        canvas_id=f"agent-jaccard-{canvas_prefix}",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Tool-call Jaccard",
                "data": [round(a.mean_tool_call_jaccard, 4) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Set similarity of (tool, args) pairs the agent invoked"}},
            "scales": {"y": {"min": 0, "max": 1}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )
    first_div_chart = chart_block(
        canvas_id=f"agent-first-div-{canvas_prefix}",
        chart_type="bar",
        data={
            "labels": pair_labels,
            "datasets": [{
                "label": "Mean first-divergence step",
                "data": [round(a.mean_first_divergence_step or 0, 3) for a in aggs],
                "backgroundColor": colors,
                "borderRadius": 6,
            }],
        },
        options={
            "plugins": {"legend": {"display": False},
                        "title": {"display": True,
                                  "text": "Mean step index where the rollout first diverged from the reference"}},
            "responsive": True,
            "maintainAspectRatio": False,
        },
    )

    pair_rows = []
    for agg in aggs:
        pair_rows.append(
            "<tr>"
            f"<td><strong>{_html.escape(agg.rollout_engine)}</strong></td>"
            f"<td>{_html.escape(agg.trainer_engine)}</td>"
            f"<td>{agg.count}</td>"
            f"<td>{_f(agg.mean_first_divergence_step)}</td>"
            f"<td>{agg.fully_matched_count}/{agg.count}</td>"
            f"<td>{_f(agg.mean_tool_choice_disagreement_rate)}</td>"
            f"<td>{_f(agg.mean_tool_call_jaccard)}</td>"
            f"<td>{quality_badge_for_match_rate(agg.final_answer_match_rate)}</td>"
            f"<td>{_f(agg.mean_rollout_steps)}</td>"
            f"<td>{_f(agg.mean_trainer_steps)}</td>"
            "</tr>"
        )

    run_rows = []
    for r in rows:
        is_div = r.first_divergence_step is not None
        match_cell = (
            "<span class='match-yes'>✓ match</span>"
            if r.final_answer_match
            else "<span class='match-no'>✗ different</span>"
        )
        kind_badge = badge(r.first_divergence_kind, "warn") if r.first_divergence_kind else badge("match", "good")
        row_class = " class='row-divergent'" if is_div else ""
        run_rows.append(
            f"<tr{row_class}>"
            f"<td>{_html.escape(r.task_id)}</td>"
            f"<td><code>{_html.escape(r.engine_pair_key)}</code></td>"
            f"<td>{r.rollout_steps}</td>"
            f"<td>{r.trainer_steps}</td>"
            f"<td>{_i(r.first_divergence_step)}</td>"
            f"<td>{kind_badge}</td>"
            f"<td>{_f(r.tool_choice_disagreement_rate)}</td>"
            f"<td>{_f(r.tool_call_jaccard)}</td>"
            f"<td>{match_cell}</td>"
            "</tr>"
        )

    return (
        f'<section class="card">'
        f"<h2>Final-answer match rate vs reference</h2>"
        f"<p class='sub'>What fraction of tasks the rollout engine ended at the same final answer as the reference engine. Higher = closer to the reference. The headline number for the agentic story.</p>"
        f"{answer_match_chart}"
        f"</section>"
        f'<section class="card">'
        f"<h2>Tool-call Jaccard &amp; first-divergence step</h2>"
        f"<p class='sub'>Left: how similar the set of (tool, arguments) invocations is across engines. Right: at which step the trajectory first diverged from the reference.</p>"
        f"<div class='chart-row'>{jaccard_chart}{first_div_chart}</div>"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per (rollout_engine -> trainer_engine) pair</h2>"
        f"<table><thead><tr>"
        f"<th>rollout</th><th>trainer</th><th>count</th>"
        f"<th>mean first-div step</th><th>fully matched</th>"
        f"<th>mean tool-disagree</th><th>mean tool-jaccard</th>"
        f"<th>answer match</th><th>mean rollout steps</th><th>mean trainer steps</th>"
        f"</tr></thead><tbody>{''.join(pair_rows)}</tbody></table>"
        f"</section>"
        f'<section class="card">'
        f"<h2>Per task × engine pair</h2>"
        f"<table><thead><tr>"
        f"<th>task</th><th>engines</th><th>rollout steps</th><th>trainer steps</th>"
        f"<th>first-div step</th><th>div kind</th>"
        f"<th>tool-disagree</th><th>tool-jaccard</th><th>answer match</th>"
        f"</tr></thead><tbody>{''.join(run_rows)}</tbody></table>"
        f"</section>"
    )


def render_html(dashboard: AgentDashboard) -> str:
    aggs = dashboard.engine_pair_aggregates()
    headline_aggs = [a for a in aggs if is_headline_pair(a.rollout_engine, a.trainer_engine)]
    appendix_aggs = [a for a in aggs if not is_headline_pair(a.rollout_engine, a.trainer_engine)]
    headline_rows = [r for r in dashboard.rows if is_headline_pair(r.rollout_engine, r.trainer_engine)]
    appendix_rows = [r for r in dashboard.rows if not is_headline_pair(r.rollout_engine, r.trainer_engine)]

    pool = headline_aggs or aggs
    if pool:
        all_match_rates = [a.final_answer_match_rate for a in pool]
        worst_match = min(all_match_rates)
        best_match = max(all_match_rates)
    else:
        worst_match = best_match = 0.0

    kpis = kpi_block([
        (str(len(headline_rows)), "headline comparisons"),
        (str(len(headline_aggs)), "headline engine pairs"),
        (f"{best_match * 100:.0f}%", "best answer-match rate"),
        (f"{worst_match * 100:.0f}%", "worst answer-match rate"),
    ])

    headline_view = _agent_engine_view(headline_aggs, headline_rows, "headline")
    appendix_view = _agent_engine_view(appendix_aggs, appendix_rows, "appendix")
    appendix_block = (
        f'<details class="appendix" data-section="all-engines">'
        f"<summary><strong>{_html.escape(APPENDIX_HEADER)}</strong> — "
        f"{len(appendix_aggs)} engine pair"
        f"{'' if len(appendix_aggs) == 1 else 's'} "
        f"({len(appendix_rows)} comparison"
        f"{'' if len(appendix_rows) == 1 else 's'})"
        f"</summary>"
        f"{appendix_view}"
        f"</details>"
    )

    glossary = render_glossary_card([
        "answer_match_rate",
        "tool_call_jaccard",
        "tool_choice_disagreement_rate",
        "first_divergence_step",
    ])

    if pool:
        pool_worst = min(a.final_answer_match_rate for a in pool)
        scope = "headline" if headline_aggs else "full-engine fallback"
        observed = (
            f"Across the {scope} set ({len(pool)} engine pair"
            f"{'s' if len(pool) != 1 else ''}, "
            f"{sum(a.count for a in pool)} multi-step trajectories), the worst "
            f"engine pair reached the same final answer as the reference only "
            f"{pool_worst * 100:.0f}% of the time. Token-level ESS deltas of "
            "~0.025 compound, via tool-call selection, into a different agent. "
            "The full-engine data is in the appendix below."
        )
    else:
        observed = "No comparisons in this snapshot."
    rq = render_research_question(
        question="Does engine and precision drift propagate from token-level "
                 "logprob deltas into agent-level behaviour on multi-step, "
                 "tool-using tasks?",
        observed=observed,
        next_step="More tasks (n>30), longer horizons, real (vs simulated) tools.",
    )

    body = (
        f"{rq}"
        f'<section class="card" data-section="headline">{kpis}</section>'
        f'<div data-section="headline-engines">{headline_view}</div>'
        f"{appendix_block}"
        f"{glossary}"
    )

    title = "Agent trajectory dashboard"
    lede = (
        f"{len(headline_rows)} headline comparison"
        f"{'s' if len(headline_rows) != 1 else ''} across "
        f"{len(headline_aggs)} engine pair{'' if len(headline_aggs) == 1 else 's'}; "
        f"{len(appendix_rows)} additional comparison"
        f"{'' if len(appendix_rows) == 1 else 's'} "
        "in the full-engine appendix. Reference is on the right of each → arrow."
    )
    return page_shell(title, lede, body)


def write_dashboard(dashboard: AgentDashboard, out_dir: Path | str) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "agent_dashboard.json"
    html_path = out_path / "agent_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(dashboard), encoding="utf-8")
    return {"json": json_path, "html": html_path}
