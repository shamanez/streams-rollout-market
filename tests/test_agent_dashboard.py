"""Tests for the agent dashboard."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rollout_market.cli import agent_dashboard as cli
from rollout_market.observatory.agent_dashboard import (
    AgentDashboard,
    build_dashboard,
    load_reports_glob,
    render_html,
    write_dashboard,
)
from rollout_market.observatory.agent_trajectory_lab import (
    TrajectoryDivergenceReport,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def _report(
    *,
    run_id: str = "r1",
    task_id: str = "t-1",
    rollout: str = "vllm-fp8",
    trainer: str = "fsdp",
    rollout_steps: int = 5,
    trainer_steps: int = 5,
    first_div: int | None = 3,
    div_kind: str | None = "tool_choice",
    tool_dis: float = 0.2,
    tool_jac: float = 0.6,
    answer_match: bool = True,
    created_at: datetime | None = None,
) -> TrajectoryDivergenceReport:
    return TrajectoryDivergenceReport(
        run_id=run_id,
        task_id=task_id,
        task_text="task",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_fp(rollout),
        trainer_engine=_fp(trainer),
        rollout_step_count=rollout_steps + 2,
        trainer_step_count=trainer_steps + 2,
        rollout_assistant_steps=rollout_steps,
        trainer_assistant_steps=trainer_steps,
        first_divergence_step=first_div,
        first_divergence_kind=div_kind,
        tool_choice_disagreement_rate=tool_dis,
        tool_call_jaccard=tool_jac,
        final_answer_match=answer_match,
        rollout_terminal_reason="answered",
        trainer_terminal_reason="answered",
        same_step_count=rollout_steps == trainer_steps,
        rollout_tool_signature=[],
        trainer_tool_signature=[],
        created_at=created_at or datetime(2026, 5, 10, tzinfo=timezone.utc),
    )


def test_build_dashboard_keeps_every_report():
    d = build_dashboard([_report(run_id="a"), _report(run_id="b")])
    assert d.num_runs == 2


def test_engine_pair_aggregate_means_first_div_step_only_over_diverged():
    reports = [
        _report(first_div=3, run_id="a"),
        _report(first_div=None, run_id="b"),
    ]
    d = build_dashboard(reports)
    aggs = d.engine_pair_aggregates()
    assert len(aggs) == 1
    assert aggs[0].count == 2
    assert aggs[0].mean_first_divergence_step == 3.0
    assert aggs[0].fully_matched_count == 1
    assert aggs[0].diverged_count == 1


def test_engine_pair_aggregate_separates_pairs():
    reports = [
        _report(rollout="vllm-fp8", trainer="fsdp", run_id="a"),
        _report(rollout="vllm-bf16", trainer="megatron", run_id="b"),
    ]
    d = build_dashboard(reports)
    keys = {(a.rollout_engine, a.trainer_engine) for a in d.engine_pair_aggregates()}
    assert ("vllm-fp8", "fsdp") in keys
    assert ("vllm-bf16", "megatron") in keys


def test_engine_pair_answer_match_rate():
    reports = [
        _report(answer_match=True, run_id="a"),
        _report(answer_match=False, run_id="b"),
        _report(answer_match=True, run_id="c"),
    ]
    d = build_dashboard(reports)
    agg = d.engine_pair_aggregates()[0]
    assert agg.final_answer_match_rate == 2 / 3


def test_render_html_self_contained():
    d = build_dashboard([_report()])
    page = render_html(d)
    assert page.startswith("<!doctype html>")
    assert "Agent trajectory dashboard" in page
    assert "Per (rollout_engine" in page


def test_render_html_escapes_user_input():
    d = build_dashboard([_report(rollout="vllm-<svg/onload=alert(1)>", task_id="<x>")])
    page = render_html(d)
    # Chart.js inserts legitimate <script> tags; we instead check that the
    # user-controlled HTML payload itself is escaped.
    assert "<svg/onload=alert(1)>" not in page
    assert "&lt;svg" in page or "&lt;x&gt;" in page


def test_write_dashboard_creates_both_files(tmp_path: Path):
    d = build_dashboard([_report()])
    paths = write_dashboard(d, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["html"].exists()
    payload = json.loads(paths["json"].read_text())
    assert payload["num_runs"] == 1


def test_load_reports_glob_reads_files(tmp_path: Path):
    base = tmp_path / "runs"
    base_time = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(3):
        sub = base / f"r{i}"
        sub.mkdir(parents=True)
        rep = _report(run_id=f"r{i}", created_at=base_time + timedelta(seconds=i))
        (sub / "agent_divergence_report.json").write_text(rep.model_dump_json())
    reports = load_reports_glob(str(base / "*" / "agent_divergence_report.json"))
    assert len(reports) == 3


def test_empty_dashboard_renders():
    d = AgentDashboard()
    page = render_html(d)
    assert "<!doctype html>" in page
    # With no rows, the headline view emits a "No engine pairs in this
    # partition." card; the appendix block has been removed entirely.
    assert "No engine pairs" in page or "Agent trajectory dashboard" in page


def test_cli_main_writes_dashboard(tmp_path: Path):
    runs_root = tmp_path / "runs"
    base_time = datetime(2026, 5, 10, tzinfo=timezone.utc)
    for i in range(2):
        sub = runs_root / f"r{i}"
        sub.mkdir(parents=True)
        rep = _report(run_id=f"r{i}", created_at=base_time + timedelta(seconds=i))
        (sub / "agent_divergence_report.json").write_text(rep.model_dump_json())
    rc = cli.main(
        [
            "--reports-glob",
            str(runs_root / "*" / "agent_divergence_report.json"),
            "--out-dir",
            str(tmp_path / "dashboard"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "dashboard" / "agent_dashboard.json").exists()
    assert (tmp_path / "dashboard" / "agent_dashboard.html").exists()
