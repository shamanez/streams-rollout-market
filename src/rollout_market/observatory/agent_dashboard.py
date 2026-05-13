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
    page_shell,
    palette_for,
    quality_badge_for_match_rate,
    raw_numbers_block,
    traffic_light_row,
    traffic_light_tile,
)
from ._device import device_bucket_label
from .agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    TrajectoryDivergenceReport,
    load_trajectory,
)


# Default search root for the trajectory viewer. The CLI / publish script
# may pass an explicit directory via ``render_html(dashboard, trajectory_dir=...)``;
# unit tests can override too. When the directory does not exist (e.g. a
# fresh checkout with no captured trajectories yet) the viewer renders an
# empty-state message instead of failing.
_DEFAULT_TRAJECTORY_DIR = Path("runs/live/agent")


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
    final_answer_match_soft: bool = False
    device_bucket: str = "L40S (g6e.12xlarge)"

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
    final_answer_match_soft_rate: float
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
                    final_answer_match_soft_rate=sum(
                        1 for r in rows if r.final_answer_match_soft
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
                    "final_answer_match_soft_rate": agg.final_answer_match_soft_rate,
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
                    "final_answer_match_soft": r.final_answer_match_soft,
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
        final_answer_match_soft=getattr(report, "final_answer_match_soft", False),
        created_at=report.created_at.isoformat(),
        device_bucket=device_bucket_label(report.device),
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
            f"<td>{_html.escape(r.device_bucket)}</td>"
            f"<td>{r.rollout_steps}</td>"
            f"<td>{r.trainer_steps}</td>"
            f"<td>{_i(r.first_divergence_step)}</td>"
            f"<td>{kind_badge}</td>"
            f"<td>{_f(r.tool_choice_disagreement_rate)}</td>"
            f"<td>{_f(r.tool_call_jaccard)}</td>"
            f"<td>{match_cell}</td>"
            "</tr>"
        )

    tl_tiles = [
        traffic_light_tile(
            label=f"{agg.rollout_engine} → {agg.trainer_engine}",
            value=agg.final_answer_match_rate,
            display=f"{agg.final_answer_match_rate * 100:.0f}%",
            sublabel=(
                f"final-answer match rate over {agg.count} task"
                f"{'' if agg.count == 1 else 's'}"
            ),
            good_at_or_above=0.66,
            bad_at_or_above=0.33,
        )
        for agg in aggs
    ]
    tl_row = traffic_light_row(tl_tiles)

    raw_tables = (
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
        f"<th>task</th><th>engines</th><th>device</th>"
        f"<th>rollout steps</th><th>trainer steps</th>"
        f"<th>first-div step</th><th>div kind</th>"
        f"<th>tool-disagree</th><th>tool-jaccard</th><th>answer match</th>"
        f"</tr></thead><tbody>{''.join(run_rows)}</tbody></table>"
        f"</section>"
    )

    return (
        f'<section class="card">'
        f"<h2>Answer-match traffic-light by engine pair</h2>"
        f"<p class='sub'>Green: ≥ 66% of tasks ended at the same final answer as the reference. Amber: 33–66%. Red: &lt; 33% (engine drift flipped most answers).</p>"
        f"{tl_row}"
        f"</section>"
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
        f"{raw_numbers_block(raw_tables)}"
    )


def _render_step(step: AgentStep) -> str:
    """Render one assistant or tool turn as an HTML block."""
    role_class = "atv-role-" + step.role
    parts = [f'<div class="atv-step {role_class}">']
    parts.append(
        f'<div class="atv-step-head">'
        f'<span class="atv-role">{_html.escape(step.role)}</span>'
        f' <span class="atv-step-idx">step #{step.step_idx}</span>'
        + (
            f' · <span class="atv-tok">{step.response_token_count} tok</span>'
            if step.response_token_count
            else ""
        )
        + (
            f' · <span class="atv-finish">{_html.escape(step.finish_reason)}</span>'
            if step.finish_reason
            else ""
        )
        + "</div>"
    )
    if step.content:
        parts.append(f'<pre class="atv-content">{_html.escape(step.content)}</pre>')
    for call in step.tool_calls:
        try:
            args_pretty = json.dumps(call.arguments, indent=2, ensure_ascii=False)
        except (TypeError, ValueError):
            args_pretty = str(call.arguments)
        parts.append(
            f'<div class="atv-toolcall">'
            f'<div class="atv-toolcall-head">→ <code>{_html.escape(call.name)}</code></div>'
            f'<pre class="atv-toolcall-args">{_html.escape(args_pretty)}</pre>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_trajectory_card(traj: AgentTrajectory) -> str:
    """One <details> block per trajectory: task header + the conversation."""
    success_badge = (
        "<span class='atv-pill atv-ok'>completed</span>"
        if traj.success
        else (
            f"<span class='atv-pill atv-term'>"
            f"{_html.escape(traj.terminal_reason or 'unfinished')}</span>"
            if traj.terminal_reason
            else "<span class='atv-pill atv-term'>unfinished</span>"
        )
    )
    n_assistant = sum(1 for s in traj.steps if s.role == "assistant")
    n_tool = sum(1 for s in traj.steps if s.role == "tool")
    summary = (
        f"<summary>"
        f"<strong>{_html.escape(traj.task_id)}</strong>"
        f" · {_html.escape(traj.model_id)}"
        f" {success_badge}"
        f" <span class='atv-meta'>"
        f"{n_assistant} assistant · {n_tool} tool · "
        f"engine <code>{_html.escape(traj.rollout_engine.name)}</code>"
        f"</span>"
        f"</summary>"
    )
    prompt_block = (
        f'<div class="atv-prompt"><div class="atv-prompt-head">task prompt</div>'
        f'<pre class="atv-content">{_html.escape(traj.task_text)}</pre></div>'
    )
    final_block = (
        f'<div class="atv-final"><div class="atv-final-head">final answer</div>'
        f'<pre class="atv-content">{_html.escape(traj.final_answer or "(none)")}</pre></div>'
    )
    steps_html = "".join(_render_step(s) for s in traj.steps)
    return (
        f'<details class="atv-trajectory" data-task-id="{_html.escape(traj.task_id)}">'
        f"{summary}{prompt_block}{steps_html}{final_block}"
        f"</details>"
    )


_TRAJECTORY_VIEWER_STYLES = """
<style>
.atv-empty{padding:1rem;color:var(--muted);font-style:italic}
.atv-trajectory{background:var(--surface);border:1px solid var(--border);
  border-radius:10px;padding:.6rem 1rem;margin:.6rem 0}
.atv-trajectory > summary{cursor:pointer;font-size:.98rem;list-style:none}
.atv-trajectory > summary::-webkit-details-marker{display:none}
.atv-trajectory[open] > summary{margin-bottom:.6rem;border-bottom:1px dashed var(--border);
  padding-bottom:.5rem}
.atv-meta{color:var(--muted);font-size:.85rem;margin-left:.4rem}
.atv-pill{display:inline-block;font-size:.72rem;padding:.05rem .45rem;border-radius:6px;
  margin-left:.3rem;vertical-align:middle}
.atv-pill.atv-ok{background:#dcfce7;color:#166534}
.atv-pill.atv-term{background:#fef3c7;color:#854d0e}
.atv-prompt,.atv-final{background:#f8fafc;border:1px solid var(--border);
  border-radius:8px;padding:.5rem .7rem;margin:.4rem 0}
.atv-prompt-head,.atv-final-head{font-size:.75rem;text-transform:uppercase;
  letter-spacing:.05em;color:var(--muted);margin-bottom:.25rem}
.atv-step{border-left:3px solid #cbd5e1;padding:.35rem .6rem;margin:.4rem 0 .4rem .2rem}
.atv-step.atv-role-assistant{border-left-color:#2563eb}
.atv-step.atv-role-tool{border-left-color:#7c3aed}
.atv-step-head{font-size:.78rem;color:var(--muted);margin-bottom:.25rem}
.atv-role{font-weight:700;text-transform:uppercase;letter-spacing:.04em;color:var(--text)}
.atv-step-idx,.atv-tok,.atv-finish{font-variant-numeric:tabular-nums}
.atv-content{white-space:pre-wrap;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.85rem;margin:0;background:transparent}
.atv-toolcall{background:#f5f3ff;border:1px solid #ddd6fe;border-radius:6px;
  padding:.4rem .6rem;margin:.35rem 0}
.atv-toolcall-head{font-size:.85rem;color:#5b21b6;margin-bottom:.2rem}
.atv-toolcall-args{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-size:.78rem;background:transparent;margin:0;white-space:pre}
</style>
""".strip()


def _list_trajectories(trajectory_dir: Path) -> list[AgentTrajectory]:
    if not trajectory_dir.is_dir():
        return []
    out: list[AgentTrajectory] = []
    for p in sorted(trajectory_dir.rglob("*.json")):
        # Skip files that don't look like a trajectory (e.g. aggregator
        # outputs) by trying load_trajectory and ignoring failures.
        try:
            out.append(load_trajectory(p))
        except Exception:
            continue
    return out


def render_html(
    dashboard: AgentDashboard,
    *,
    trajectory_dir: Path | str | None = None,
) -> str:
    """Render the agent-trajectory page as a plain *viewer*.

    Per operator direction (2026-05-13), this page is no longer a
    rollout-vs-trainer comparison dashboard. The trainer only ever sees
    the tokens the rollout generated, so per-token / sequence-level
    mismatch is the relevant story and is already covered by the dense
    + router matrices on the index. This page just lets a reader page
    through the actual captured trajectories — prompt, response tokens,
    tool calls embedded in the response. No engine comparison, no
    answer-match, no Jaccard, no first-divergence-step.

    The ``dashboard`` parameter is retained so the surrounding CLI
    (which calls ``build_dashboard(...)`` first) still works; its
    contents are not displayed.
    """
    _ = dashboard  # intentionally unused on this page
    traj_dir = Path(trajectory_dir) if trajectory_dir is not None else _DEFAULT_TRAJECTORY_DIR
    trajectories = _list_trajectories(traj_dir)
    if trajectories:
        cards = "".join(_render_trajectory_card(t) for t in trajectories)
        body_inner = (
            '<section class="card" data-section="trajectories">'
            f"<h2>Captured trajectories ({len(trajectories)})</h2>"
            "<p class='sub'>Each block shows one task's prompt, the response tokens "
            "the rollout produced, and the tool calls embedded in the response. "
            "This page is a viewer; the training-inference mismatch on these tokens "
            "is on the Dense and Router matrices on the home page.</p>"
            f"{cards}"
            "</section>"
        )
    else:
        body_inner = (
            '<section class="card" data-section="trajectories">'
            "<h2>Captured trajectories (0)</h2>"
            "<p class='atv-empty'>No trajectories found under "
            f"<code>{_html.escape(str(traj_dir))}</code>. Run a capture pass (see "
            "<code>CLAUDE.md</code> § End-to-end reproduction) to populate.</p>"
            "</section>"
        )
    body = f"{_TRAJECTORY_VIEWER_STYLES}{body_inner}"
    title = "Agent trajectory viewer"
    lede = (
        "Plain viewer for captured Hermes-Agent trajectories. "
        "Comparison metrics moved to the Dense and Router matrices on the home page."
    )
    return page_shell(title, lede, body)


def write_dashboard(
    dashboard: AgentDashboard,
    out_dir: Path | str,
    *,
    trajectory_dir: Path | str | None = None,
) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "agent_dashboard.json"
    html_path = out_path / "agent_dashboard.html"
    json_path.write_text(json.dumps(dashboard.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(
        render_html(dashboard, trajectory_dir=trajectory_dir), encoding="utf-8"
    )
    return {"json": json_path, "html": html_path}
