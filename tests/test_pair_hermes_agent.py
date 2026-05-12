"""Tests for scripts/live/pair_hermes_agent.py replicate-aware pairing.

The n=30 follow-up needs multiple replicates per side so the noise-floor
and precision-effect tiles get tighter bars. The pair script now accepts
multiple trajectory dirs per side (repeat the flag) and produces the
full cartesian-product of (rollout replicate, trainer replicate) pairs
per task.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = REPO_ROOT / "scripts" / "live" / "pair_hermes_agent.py"
_spec = importlib.util.spec_from_file_location("pair_hermes_agent", _SCRIPT)
pair_hermes_agent = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pair_hermes_agent)  # type: ignore[union-attr]


def _make_traj(*, task_id: str, engine: str, final: str = "x", n_steps: int = 2) -> AgentTrajectory:
    steps = [
        AgentStep(
            step_idx=0,
            role="assistant",
            content="",
            tool_calls=[ToolCallRecord.from_call(name="calculator", arguments={}, result="r")],
        )
    ]
    if n_steps > 1:
        steps.append(AgentStep(step_idx=0, role="tool", content="r"))
    if n_steps > 2:
        steps.append(AgentStep(step_idx=1, role="assistant", content=final))
    return AgentTrajectory(
        run_id=f"{engine}-{task_id}",
        task_id=task_id,
        task_text="t",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=EngineFingerprint(name=engine, version="0.0", fingerprint=f"sha256:{engine}"),
        available_tools=[],
        steps=steps,
        terminal_reason="answered",
        final_answer=final,
    )


def _write(path: Path, traj: AgentTrajectory) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")


def test_list_trajectories_buckets_replicates_per_task(tmp_path: Path) -> None:
    """Two replicate dirs for the same engine should collect into one
    list keyed by task_id."""
    rep1 = tmp_path / "bf16_rep1"
    rep2 = tmp_path / "bf16_rep2"
    _write(rep1 / "t-1.json", _make_traj(task_id="t-1", engine="bf16-rep1"))
    _write(rep1 / "t-2.json", _make_traj(task_id="t-2", engine="bf16-rep1"))
    _write(rep2 / "t-1.json", _make_traj(task_id="t-1", engine="bf16-rep2"))
    out = pair_hermes_agent._list_trajectories([rep1, rep2])
    assert set(out.keys()) == {"t-1", "t-2"}
    assert len(out["t-1"]) == 2  # one from each replicate
    assert len(out["t-2"]) == 1
    # File paths point at the right dirs.
    rep1_str = str(rep1)
    rep2_str = str(rep2)
    assert any(rep1_str in str(p) for p in out["t-1"])
    assert any(rep2_str in str(p) for p in out["t-1"])


def test_list_trajectories_handles_missing_dirs(tmp_path: Path) -> None:
    """Missing or empty dirs are skipped silently."""
    real = tmp_path / "real"
    _write(real / "t-1.json", _make_traj(task_id="t-1", engine="bf16"))
    out = pair_hermes_agent._list_trajectories(
        [real, tmp_path / "does-not-exist", tmp_path / "also-missing"]
    )
    assert list(out.keys()) == ["t-1"]


def test_pair_side_single_replicate_each(tmp_path: Path, monkeypatch) -> None:
    """One replicate per side -> one pair per shared task (original behaviour)."""
    rollout = tmp_path / "fp8"
    trainer = tmp_path / "bf16"
    _write(rollout / "t-1.json", _make_traj(task_id="t-1", engine="fp8"))
    _write(rollout / "t-2.json", _make_traj(task_id="t-2", engine="fp8"))
    _write(trainer / "t-1.json", _make_traj(task_id="t-1", engine="bf16"))
    _write(trainer / "t-2.json", _make_traj(task_id="t-2", engine="bf16"))

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        pair_hermes_agent, "_pair_one", lambda r, t, out_root: calls.append((r, t))
    )
    n = pair_hermes_agent._pair_side(
        rollout_dirs=[rollout], trainer_dirs=[trainer], out_root=tmp_path / "out"
    )
    assert n == 2
    assert len(calls) == 2


def test_pair_side_replicates_produce_cartesian_product(tmp_path: Path, monkeypatch) -> None:
    """N rollout replicates x M trainer replicates per task -> N*M pairs each."""
    fp8_rep1 = tmp_path / "fp8_rep1"
    fp8_rep2 = tmp_path / "fp8_rep2"
    bf16_rep1 = tmp_path / "bf16_rep1"
    bf16_rep2 = tmp_path / "bf16_rep2"
    bf16_rep3 = tmp_path / "bf16_rep3"
    for d in (fp8_rep1, fp8_rep2):
        _write(d / "t-1.json", _make_traj(task_id="t-1", engine="fp8"))
    for d in (bf16_rep1, bf16_rep2, bf16_rep3):
        _write(d / "t-1.json", _make_traj(task_id="t-1", engine="bf16"))

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        pair_hermes_agent, "_pair_one", lambda r, t, out_root: calls.append((r, t))
    )
    n = pair_hermes_agent._pair_side(
        rollout_dirs=[fp8_rep1, fp8_rep2],
        trainer_dirs=[bf16_rep1, bf16_rep2, bf16_rep3],
        out_root=tmp_path / "out",
    )
    # 2 fp8 replicates x 3 bf16 replicates x 1 shared task = 6 pairs.
    assert n == 6
    assert len(calls) == 6


def test_pair_side_skips_self_pair(tmp_path: Path, monkeypatch) -> None:
    """When rollout and trainer share a directory (noise-floor self-compare),
    the script must not pair a trajectory against itself."""
    d = tmp_path / "bf16"
    _write(d / "t-1.json", _make_traj(task_id="t-1", engine="bf16"))
    _write(d / "t-2.json", _make_traj(task_id="t-2", engine="bf16"))

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        pair_hermes_agent, "_pair_one", lambda r, t, out_root: calls.append((r, t))
    )
    n = pair_hermes_agent._pair_side(
        rollout_dirs=[d], trainer_dirs=[d], out_root=tmp_path / "out"
    )
    # Both files share the same dir -> r == t for every task -> no pairs.
    assert n == 0


def test_pair_side_only_intersects_shared_task_ids(tmp_path: Path, monkeypatch) -> None:
    """Tasks present on only one side must not produce pairs."""
    rollout = tmp_path / "fp8"
    trainer = tmp_path / "bf16"
    _write(rollout / "shared.json", _make_traj(task_id="shared", engine="fp8"))
    _write(rollout / "rollout-only.json", _make_traj(task_id="rollout-only", engine="fp8"))
    _write(trainer / "shared.json", _make_traj(task_id="shared", engine="bf16"))
    _write(trainer / "trainer-only.json", _make_traj(task_id="trainer-only", engine="bf16"))

    calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        pair_hermes_agent, "_pair_one", lambda r, t, out_root: calls.append((r, t))
    )
    n = pair_hermes_agent._pair_side(
        rollout_dirs=[rollout], trainer_dirs=[trainer], out_root=tmp_path / "out"
    )
    assert n == 1
    assert "shared.json" in str(calls[0][0])
    assert "shared.json" in str(calls[0][1])


def test_pair_side_empty_dirs_return_zero(tmp_path: Path) -> None:
    """Both dirs empty -> zero pairs, no error."""
    rollout = tmp_path / "fp8"
    trainer = tmp_path / "bf16"
    rollout.mkdir()
    trainer.mkdir()
    n = pair_hermes_agent._pair_side(
        rollout_dirs=[rollout], trainer_dirs=[trainer], out_root=tmp_path / "out"
    )
    assert n == 0


def test_cli_accepts_multiple_bf16_dir_flags() -> None:
    """The --bf16-dir flag is repeatable; argparse collects into a list."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bf16-dir", action="append", required=True)
    p.add_argument("--model-slug", required=True)
    ns = p.parse_args(["--model-slug", "x", "--bf16-dir", "a", "--bf16-dir", "b"])
    assert ns.bf16_dir == ["a", "b"]


def test_main_falls_back_when_no_replicates_provided(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI entry point: one --bf16-dir + one --fp8-dir works exactly like before."""
    rollout = tmp_path / "fp8"
    trainer = tmp_path / "bf16"
    _write(rollout / "t-1.json", _make_traj(task_id="t-1", engine="fp8"))
    _write(trainer / "t-1.json", _make_traj(task_id="t-1", engine="bf16"))
    monkeypatch.setattr(
        pair_hermes_agent, "_pair_one", lambda r, t, out_root: None
    )
    # Skip the aggregator subprocess.
    monkeypatch.setattr(
        pair_hermes_agent.subprocess, "run", lambda *a, **k: None
    )
    rc = pair_hermes_agent.main(
        [
            "--model-slug", "qwen3-32b",
            "--bf16-dir", str(trainer),
            "--fp8-dir", str(rollout),
            "--out-base", str(tmp_path / "out"),
            "--dashboard-out", str(tmp_path / "dash"),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "fp8-vs-bf16: 1 pairs" in captured
