"""Offline tests for scripts/live/pair_hermes_moe_reports.py.

Synthetic index + trainer routed_experts + AgentTrajectory fixtures
exercise the MoE pair script without touching the spot or any real
trainer/router code.

Tests cover:

  * happy path — matched rollout/trainer expert_ids → 0 flip rate.
  * divergent expert_ids → measurable router_flip_rate.
  * missing response_routed_experts on the step → ValueError.
  * unknown trajectory_run_id → ValueError.
  * trainer_engine_label override propagates.
  * CLI round-trip writes per-(trajectory, turn) subdirs.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest

from rollout_market.observatory.agent_trajectory_lab import AgentStep, AgentTrajectory
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint
from rollout_market.observatory.router_mismatch_lab import RouterMismatchReport


def _load_module() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "pair_hermes_moe_reports.py"
    )
    spec = importlib.util.spec_from_file_location("pair_hermes_moe_reports", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pair_mod() -> ModuleType:
    return _load_module()


# Small MoE shape: 3 tokens × 2 layers × top-1 expert from a 4-expert pool.
_NUM_LAYERS = 2
_NUM_EXPERTS = 4
_TOP_K = 1


def _moe_trajectory(
    *,
    task_id: str = "moe-task",
    routed_experts: list[list[list[int]]] | None = None,
) -> AgentTrajectory:
    if routed_experts is None:
        routed_experts = [[[0], [1]], [[1], [2]], [[2], [3]]]
    step = AgentStep(
        step_idx=0,
        role="assistant",
        content="answer",
        prompt_token_ids=[10, 11, 12, 13],
        response_token_ids=[100, 101, 102],
        response_logprobs=[-0.1, -0.2, -0.3],
        response_routed_experts=routed_experts,
    )
    return AgentTrajectory(
        run_id=f"hermes-qwen3-30b-a3b-bf16-{task_id}",
        task_id=task_id,
        task_text="q",
        model_id="Qwen/Qwen3-30B-A3B",
        rollout_engine=EngineFingerprint(
            name="hermes-qwen3-30b-a3b-bf16",
            version="0.20.1",
            fingerprint="sha256:hermes-qwen3-30b-a3b-bf16-tp2-l40s",
        ),
        available_tools=[],
        steps=[step],
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
        created_at=datetime.now(timezone.utc),
    )


def _make_index(traj: AgentTrajectory) -> dict:
    return {
        "model_id": traj.model_id,
        "precision_class": "bf16",
        "entries": [
            {
                "prompt_idx": 0,
                "task_id": traj.task_id,
                "trajectory_run_id": traj.run_id,
                "assistant_turn_idx": 0,
                "num_tokens": 3,
                "response_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
        "skipped": [],
    }


def _make_trainers(expert_ids: list[list[list[int]]]) -> list[dict]:
    return [
        {
            "prompt_idx": 0,
            "model": "Qwen/Qwen3-30B-A3B",
            "engine": "fsdp",
            "engine_fingerprint": "sha256:fsdp-moe-bf16",
            "num_layers": _NUM_LAYERS,
            "num_experts": _NUM_EXPERTS,
            "top_k": _TOP_K,
            "expert_ids": expert_ids,
            "world_size": 4,
        }
    ]


def test_pair_emits_one_report_per_turn(pair_mod: ModuleType) -> None:
    traj = _moe_trajectory()
    index = _make_index(traj)
    # Matched expert ids → zero flip rate.
    trainers = _make_trainers([[[0], [1]], [[1], [2]], [[2], [3]]])
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
    )
    assert len(reports) == 1
    rep = reports[0]
    assert isinstance(rep, RouterMismatchReport)
    assert rep.rollout_engine.name == "hermes-qwen3-30b-a3b-bf16"
    assert rep.trainer_engine.name == "fsdp"
    assert rep.num_tokens == 3
    assert rep.num_layers == _NUM_LAYERS
    assert rep.router_flip_rate == 0.0


def test_pair_surfaces_router_flip_rate_on_divergence(pair_mod: ModuleType) -> None:
    """Distinct rollout vs trainer expert_ids must produce a non-zero
    router_flip_rate end-to-end."""
    traj = _moe_trajectory(routed_experts=[[[0], [1]], [[1], [2]], [[2], [3]]])
    index = _make_index(traj)
    # Disagree on the FIRST layer for every token → layer-0 flip
    # rate 100%, layer-1 0% → overall 50%.
    trainers = _make_trainers([[[3], [1]], [[3], [2]], [[3], [3]]])
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
    )
    rep = reports[0]
    assert rep.router_flip_rate == pytest.approx(0.5)
    assert rep.layer_flip_rates[0] == pytest.approx(1.0)
    assert rep.layer_flip_rates[1] == pytest.approx(0.0)


def test_pair_raises_on_missing_routed_experts(pair_mod: ModuleType) -> None:
    """A step that lacks captured routed_experts cannot be paired."""
    # Build a trajectory whose step has response_token_ids / logprobs
    # but no routed_experts.
    step = AgentStep(
        step_idx=0,
        role="assistant",
        content="answer",
        prompt_token_ids=[10, 11, 12],
        response_token_ids=[100, 101, 102],
        response_logprobs=[-0.1, -0.2, -0.3],
    )
    traj = AgentTrajectory(
        run_id="hermes-noroute-x",
        task_id="x",
        task_text="q",
        model_id="Qwen/Qwen3-30B-A3B",
        rollout_engine=EngineFingerprint(
            name="hermes-qwen3-30b-a3b-bf16",
            version="0.20.1",
            fingerprint="sha256:x",
        ),
        available_tools=[],
        steps=[step],
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
    )
    index = _make_index(traj)
    trainers = _make_trainers([[[0], [1]], [[1], [2]], [[2], [3]]])
    with pytest.raises(ValueError, match="response_routed_experts"):
        pair_mod.pair_reports(
            index=index,
            trainers=trainers,
            trajectories_by_run_id={traj.run_id: traj},
            precision_class="bf16",
        )


def test_pair_raises_on_unknown_trajectory(pair_mod: ModuleType) -> None:
    traj = _moe_trajectory()
    index = _make_index(traj)
    trainers = _make_trainers([[[0], [1]], [[1], [2]], [[2], [3]]])
    with pytest.raises(ValueError, match="trajectory_run_id"):
        pair_mod.pair_reports(
            index=index,
            trainers=trainers,
            trajectories_by_run_id={},
            precision_class="bf16",
        )


def test_pair_trainer_engine_label_override(pair_mod: ModuleType) -> None:
    traj = _moe_trajectory()
    index = _make_index(traj)
    trainers = _make_trainers([[[0], [1]], [[1], [2]], [[2], [3]]])
    reports = pair_mod.pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id={traj.run_id: traj},
        precision_class="bf16",
        trainer_engine_label="megatron-lm",
    )
    assert reports[0].trainer_engine.name == "megatron-lm"


def test_cli_writes_reports_under_out_root(pair_mod: ModuleType, tmp_path: Path) -> None:
    traj = _moe_trajectory()
    index = _make_index(traj)
    trainers = _make_trainers([[[0], [1]], [[1], [2]], [[2], [3]]])
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    (traj_dir / "moe-task.json").write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    index_path = tmp_path / "index.json"
    trainers_path = tmp_path / "trainers.json"
    out_root = tmp_path / "out"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    trainers_path.write_text(json.dumps(trainers), encoding="utf-8")

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
    written = sorted(out_root.rglob("router_mismatch_report.json"))
    assert len(written) == 1
    parsed = RouterMismatchReport.model_validate_json(written[0].read_text())
    assert parsed.trainer_engine.name == "fsdp-bf16"
    assert parsed.num_tokens == 3
