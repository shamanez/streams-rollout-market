"""Offline tests for scripts/live/pair_hermes_dense_reports.py.

Synthetic ``index`` + ``trainers`` payloads + an ``AgentTrajectory``
fixture exercise the joining logic without touching the spot or the
real trainer-side code. Tests cover:

  * happy path — one trajectory, two turns, one report per turn with
    matching ``prompt_id`` and the rollout/trainer engine labels.
  * length-mismatch surfaces as a ``ValueError`` mentioning the
    offending prompt_idx.
  * missing trainer entry surfaces as a ``ValueError``.
  * trainer_engine_label override propagates to ``trainer_engine.name``.
  * CLI round-trip writes per-(task, turn) directories under
    ``out-root`` and the rendered JSON parses back into a
    ``DenseMismatchReport``.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
)
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
)


def _load_module() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "pair_hermes_dense_reports.py"
    )
    spec = importlib.util.spec_from_file_location("pair_hermes_dense_reports", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pair_mod() -> ModuleType:
    return _load_module()


def _make_trajectory(task_id: str, n_turns: int = 2) -> AgentTrajectory:
    steps: list[AgentStep] = []
    for i in range(n_turns):
        steps.append(
            AgentStep(
                step_idx=i,
                role="assistant",
                content=f"turn {i}",
                prompt_token_ids=list(range(10, 10 + 3 * (i + 1))),
                response_token_ids=[100 + i * 10, 101 + i * 10, 102 + i * 10],
                response_logprobs=[-0.1, -0.2, -0.3],
            )
        )
    return AgentTrajectory(
        run_id=f"hermes-qwen3-32b-bf16-{task_id}",
        task_id=task_id,
        task_text="x",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=EngineFingerprint(
            name="hermes-qwen3-32b-bf16",
            version="0.20.1",
            fingerprint="sha256:hermes-qwen3-32b-bf16-tp4-l40s",
        ),
        available_tools=[],
        steps=steps,
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
        created_at=datetime.now(timezone.utc),
    )


def _make_index_and_trainers(
    traj: AgentTrajectory,
    *,
    trainer_logprobs: list[list[float]] | None = None,
) -> tuple[dict, list[dict]]:
    """Build matched index + trainers payloads from a trajectory."""
    entries: list[dict] = []
    trainers: list[dict] = []
    assistant_steps = [s for s in traj.steps if s.role == "assistant"]
    for i, step in enumerate(assistant_steps):
        assert step.response_logprobs is not None
        assert step.response_token_ids is not None
        entries.append(
            {
                "prompt_idx": i,
                "task_id": traj.task_id,
                "trajectory_run_id": traj.run_id,
                "assistant_turn_idx": i,
                "num_tokens": len(step.response_token_ids),
                "response_logprobs": list(step.response_logprobs),
            }
        )
        tlp = (
            trainer_logprobs[i]
            if trainer_logprobs is not None
            else [lp + 0.01 for lp in step.response_logprobs]
        )
        trainers.append(
            {
                "model": traj.model_id,
                "trainer_logprobs": tlp,
                "engine": "fsdp",
                "attn_impl": "sdpa",
                "dtype": "bfloat16",
                "world_size": 4,
                "sharding": "FULL_SHARD",
                "prompt_idx": i,
            }
        )
    index = {
        "model_id": traj.model_id,
        "precision_class": "bf16",
        "entries": entries,
        "skipped": [],
    }
    return index, trainers


def test_pair_emits_one_report_per_turn(pair_mod: ModuleType) -> None:
    traj = _make_trajectory("t1", n_turns=2)
    index, trainers = _make_index_and_trainers(traj)
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
    )
    assert len(reports) == 2
    assert all(isinstance(r, DenseMismatchReport) for r in reports)
    assert [r.prompt_id for r in reports] == ["t1-turn0", "t1-turn1"]
    # Rollout engine carries through from the trajectory; trainer
    # engine comes from the trainers payload.
    assert reports[0].rollout_engine.name == "hermes-qwen3-32b-bf16"
    assert reports[0].trainer_engine.name == "fsdp"
    assert reports[0].precision_class == "bf16"
    assert reports[0].num_policy_tokens == 3


def test_pair_uses_trainer_engine_label_override(pair_mod: ModuleType) -> None:
    traj = _make_trajectory("t1", n_turns=1)
    index, trainers = _make_index_and_trainers(traj)
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
        trainer_engine_label="fsdp-bf16",
    )
    assert reports[0].trainer_engine.name == "fsdp-bf16"


def test_pair_raises_on_length_mismatch(pair_mod: ModuleType) -> None:
    traj = _make_trajectory("t1", n_turns=1)
    index, trainers = _make_index_and_trainers(traj, trainer_logprobs=[[-0.1, -0.2]])
    with pytest.raises(ValueError, match="length mismatch"):
        pair_mod.pair_reports(
            index=index,
            trainers=trainers,
            trajectories_by_run_id={traj.run_id: traj},
            precision_class="bf16",
        )


def test_pair_raises_on_missing_trainer_entry(pair_mod: ModuleType) -> None:
    traj = _make_trajectory("t1", n_turns=2)
    index, trainers = _make_index_and_trainers(traj)
    trainers = [trainers[0]]  # drop the second turn's trainer entry
    with pytest.raises(ValueError, match="trainer entry missing"):
        pair_mod.pair_reports(
            index=index,
            trainers=trainers,
            trajectories_by_run_id={traj.run_id: traj},
            precision_class="bf16",
        )


def test_pair_raises_on_unknown_trajectory(pair_mod: ModuleType) -> None:
    traj = _make_trajectory("t1", n_turns=1)
    index, trainers = _make_index_and_trainers(traj)
    with pytest.raises(ValueError, match="trajectory_run_id"):
        pair_mod.pair_reports(
            index=index,
            trainers=trainers,
            trajectories_by_run_id={},
            precision_class="bf16",
        )


def test_pair_response_logprobs_drive_ess(pair_mod: ModuleType) -> None:
    """Equal rollout and trainer logprobs must yield ESS == 1.0; that's
    the load-bearing math contract."""
    traj = _make_trajectory("t1", n_turns=1)
    index, _ = _make_index_and_trainers(traj)
    matched_trainers = [
        {
            "trainer_logprobs": list(index["entries"][0]["response_logprobs"]),
            "engine": "fsdp",
            "sharding": "FULL_SHARD",
            "dtype": "bfloat16",
            "world_size": 4,
            "prompt_idx": 0,
        }
    ]
    reports = pair_mod.pair_reports(
        index=index,
        trainers=matched_trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
    )
    assert reports[0].ess == pytest.approx(1.0)
    assert reports[0].delta_logprob_mean == pytest.approx(0.0)
    assert reports[0].delta_logprob_abs_mean == pytest.approx(0.0)


def test_pair_handles_megatron_shaped_trainer_payload(pair_mod: ModuleType) -> None:
    """The pair script's engine-fingerprint helper must build a sensible
    EngineFingerprint from a Megatron-shaped trainer entry (engine=
    'megatron-lm', no FSDP-specific keys). Same recipe as the FSDP path
    but verifies the engine label survives all the way to the rendered
    report's trainer_engine.name."""
    traj = _make_trajectory("t1", n_turns=1)
    index, _ = _make_index_and_trainers(traj)
    megatron_trainers = [
        {
            "trainer_logprobs": list(index["entries"][0]["response_logprobs"]),
            "engine": "megatron-lm",
            "engine_fingerprint": "sha256:megatron-core-r0.13-tp4-bf16-torch-dist",
            "dtype": "bfloat16",
            "world_size": 4,
            "prompt_idx": 0,
        }
    ]
    reports = pair_mod.pair_reports(
        index=index,
        trainers=megatron_trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
    )
    rep = reports[0]
    # The trainer engine name propagates from the trainer entry.
    assert rep.trainer_engine.name == "megatron-lm"
    # And the explicit engine_fingerprint from the trainer payload
    # survives onto the report (it identifies the exact trainer config).
    assert "megatron-core" in rep.trainer_engine.fingerprint
    # An explicit label override beats the trainer-entry value.
    reports_override = pair_mod.pair_reports(
        index=index,
        trainers=megatron_trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
        trainer_engine_label="megatron-bf16",
    )
    assert reports_override[0].trainer_engine.name == "megatron-bf16"


def test_cli_writes_reports_under_out_root(pair_mod: ModuleType, tmp_path: Path) -> None:
    traj = _make_trajectory("t1", n_turns=2)
    index, trainers = _make_index_and_trainers(traj)
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    (traj_dir / "t1.json").write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    index_path = tmp_path / "index.json"
    trainers_path = tmp_path / "trainers.json"
    out_root = tmp_path / "out"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    trainers_path.write_text(json.dumps(trainers, indent=2), encoding="utf-8")

    rc = pair_mod.main(
        [
            "--index",
            str(index_path),
            "--trainers",
            str(trainers_path),
            "--trajectory-dir",
            str(traj_dir),
            "--out-root",
            str(out_root),
            "--trainer-engine-label",
            "fsdp-bf16",
        ]
    )
    assert rc == 0
    turn0 = out_root / "t1-turn0" / "dense_mismatch_report.json"
    turn1 = out_root / "t1-turn1" / "dense_mismatch_report.json"
    assert turn0.exists()
    assert turn1.exists()
    # Re-parse and sanity-check the rendered report.
    parsed = DenseMismatchReport.model_validate_json(turn0.read_text())
    assert parsed.trainer_engine.name == "fsdp-bf16"
    assert parsed.prompt_id == "t1-turn0"
    assert parsed.precision_class == "bf16"
