"""Offline tests for scripts/live/build_hermes_dense_input.py.

The script flattens Hermes-Agent ``AgentTrajectory`` JSONs into the
``/tmp/rollouts.json`` shape consumed by ``run_fsdp_reference.py``,
plus a sidecar index file that lets the downstream pairing script
join the trainer's outputs back to the originating (trajectory, turn).

Tests cover:

  * happy path — multi-turn trajectory with probes on every assistant
    step produces N entries with stable ``prompt_idx`` and matching
    index records.
  * partial probes — turns missing ``prompt_token_ids`` /
    ``response_token_ids`` / ``response_logprobs`` are skipped and
    listed in ``index['skipped']`` with a non-empty reason.
  * model_id consistency — disagreeing model_ids across trajectories
    raises (unless an override is passed).
  * CLI round-trip — running ``main()`` end-to-end against a fixture
    directory writes a valid rollouts.json + index.json pair.
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
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint


def _load_module() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "build_hermes_dense_input.py"
    )
    spec = importlib.util.spec_from_file_location("build_hermes_dense_input", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_module()


def _engine() -> EngineFingerprint:
    return EngineFingerprint(
        name="hermes-qwen3-32b-bf16",
        version="0.20.1",
        fingerprint="sha256:hermes-qwen3-32b-bf16-tp4-l40s",
    )


def _trajectory_with_probes(
    *,
    task_id: str,
    model_id: str = "Qwen/Qwen3-32B",
    n_turns: int = 2,
    drop_probes_on_turn: int | None = None,
    drop_logprobs_on_turn: int | None = None,
) -> AgentTrajectory:
    """Build a synthetic multi-turn trajectory; optionally drop probes on a turn."""
    steps: list[AgentStep] = []
    for i in range(n_turns):
        if i == drop_probes_on_turn:
            steps.append(
                AgentStep(
                    step_idx=i,
                    role="assistant",
                    content=f"turn {i} (no probes)",
                )
            )
        elif i == drop_logprobs_on_turn:
            steps.append(
                AgentStep(
                    step_idx=i,
                    role="assistant",
                    content=f"turn {i} (no logprobs)",
                    prompt_token_ids=[10, 11, 12],
                    response_token_ids=[20, 21, 22],
                )
            )
        else:
            base_prompt = list(range(10, 10 + 3 * (i + 1)))
            steps.append(
                AgentStep(
                    step_idx=i,
                    role="assistant",
                    content=f"turn {i}",
                    prompt_token_ids=base_prompt,
                    response_token_ids=[100 + i * 10, 101 + i * 10, 102 + i * 10],
                    response_logprobs=[-0.1 - i * 0.01, -0.2 - i * 0.01, -0.3 - i * 0.01],
                )
            )
    return AgentTrajectory(
        run_id=f"hermes-qwen3-32b-bf16-{task_id}",
        task_id=task_id,
        task_text=f"Task {task_id}",
        model_id=model_id,
        rollout_engine=_engine(),
        available_tools=[],
        steps=steps,
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
        created_at=datetime.now(timezone.utc),
    )


def test_build_happy_path_assigns_sequential_prompt_idx(builder: ModuleType) -> None:
    trajectories = [
        _trajectory_with_probes(task_id="t1", n_turns=2),
        _trajectory_with_probes(task_id="t2", n_turns=3),
    ]
    rollouts, index = builder.build_rollouts_and_index(trajectories, precision_class="bf16")
    # 2 + 3 turns = 5 entries.
    assert len(rollouts) == 5
    assert [r["prompt_idx"] for r in rollouts] == [0, 1, 2, 3, 4]
    assert all(r["model"] == "Qwen/Qwen3-32B" for r in rollouts)
    assert all(r["trainer_reference"] == "Qwen/Qwen3-32B" for r in rollouts)
    # Index entries align with rollouts.
    assert [e["prompt_idx"] for e in index["entries"]] == [0, 1, 2, 3, 4]
    assert index["skipped"] == []
    assert index["model_id"] == "Qwen/Qwen3-32B"
    assert index["precision_class"] == "bf16"


def test_build_preserves_response_logprobs_in_index(builder: ModuleType) -> None:
    traj = _trajectory_with_probes(task_id="t1", n_turns=1)
    _, index = builder.build_rollouts_and_index([traj], precision_class="bf16")
    assert len(index["entries"]) == 1
    entry = index["entries"][0]
    assert entry["task_id"] == "t1"
    assert entry["assistant_turn_idx"] == 0
    assert entry["num_tokens"] == 3
    assert entry["response_logprobs"] == [-0.1, -0.2, -0.3]


def test_build_skips_turn_missing_token_ids(builder: ModuleType) -> None:
    traj = _trajectory_with_probes(task_id="t1", n_turns=3, drop_probes_on_turn=1)
    rollouts, index = builder.build_rollouts_and_index([traj], precision_class="bf16")
    # Turns 0 and 2 made it through.
    assert len(rollouts) == 2
    assert [e["assistant_turn_idx"] for e in index["entries"]] == [0, 2]
    assert len(index["skipped"]) == 1
    skip = index["skipped"][0]
    assert skip["assistant_turn_idx"] == 1
    assert "prompt_token_ids" in skip["reason"] or "response_token_ids" in skip["reason"]


def test_build_skips_turn_missing_logprobs(builder: ModuleType) -> None:
    """Turns with token IDs but no rollout-side logprobs have nothing
    to compare against on the rollout side; they must be skipped."""
    traj = _trajectory_with_probes(task_id="t1", n_turns=2, drop_logprobs_on_turn=1)
    rollouts, index = builder.build_rollouts_and_index([traj], precision_class="bf16")
    assert len(rollouts) == 1
    assert rollouts[0]["assistant_turn_idx"] == 0
    assert len(index["skipped"]) == 1
    assert "response_logprobs" in index["skipped"][0]["reason"]


def test_build_rejects_disagreeing_model_ids(builder: ModuleType) -> None:
    """Mixing trajectories from different models on one trainer-side run
    would silently mismatch tokenizer hashes; abort instead."""
    trajectories = [
        _trajectory_with_probes(task_id="t1", model_id="Qwen/Qwen3-32B"),
        _trajectory_with_probes(task_id="t2", model_id="Qwen/Qwen3-30B-A3B"),
    ]
    with pytest.raises(ValueError, match="model_id"):
        builder.build_rollouts_and_index(trajectories, precision_class="bf16")


def test_build_model_id_override_unifies(builder: ModuleType) -> None:
    """The override forces a single trainer-side model id even when
    the input trajectories disagree."""
    trajectories = [
        _trajectory_with_probes(task_id="t1", model_id="Qwen/Qwen3-32B"),
        _trajectory_with_probes(task_id="t2", model_id="Qwen/Qwen3-30B-A3B"),
    ]
    rollouts, index = builder.build_rollouts_and_index(
        trajectories,
        precision_class="bf16",
        model_id_override="Qwen/Qwen3-32B",
    )
    assert all(r["model"] == "Qwen/Qwen3-32B" for r in rollouts)
    assert index["model_id"] == "Qwen/Qwen3-32B"


def test_build_tool_steps_are_ignored(builder: ModuleType) -> None:
    """Tool-role steps don't generate model tokens; they must not produce
    rollout entries (and the validator already forbids probes on them)."""
    traj = AgentTrajectory(
        run_id="r1",
        task_id="t1",
        task_text="x",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_engine(),
        available_tools=[],
        steps=[
            AgentStep(
                step_idx=0,
                role="assistant",
                content="call tool",
                prompt_token_ids=[1, 2, 3],
                response_token_ids=[4, 5],
                response_logprobs=[-0.1, -0.2],
            ),
            AgentStep(step_idx=0, role="tool", content="result"),
            AgentStep(
                step_idx=1,
                role="assistant",
                content="answer",
                prompt_token_ids=[1, 2, 3, 4, 5, 6],
                response_token_ids=[7, 8, 9],
                response_logprobs=[-0.05, -0.1, -0.15],
            ),
        ],
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="answer",
    )
    rollouts, _ = builder.build_rollouts_and_index([traj], precision_class="bf16")
    # Only the two assistant turns; the tool step contributes nothing.
    assert len(rollouts) == 2
    assert rollouts[0]["response_token_ids"] == [4, 5]
    assert rollouts[1]["response_token_ids"] == [7, 8, 9]


def test_cli_writes_rollouts_and_index_files(builder: ModuleType, tmp_path: Path) -> None:
    """End-to-end: the CLI should walk a trajectory dir and write both
    output files in the expected shape."""
    in_dir = tmp_path / "traj"
    in_dir.mkdir()
    for i, task_id in enumerate(["alpha", "beta"]):
        traj = _trajectory_with_probes(task_id=task_id, n_turns=i + 1)
        (in_dir / f"{task_id}.json").write_text(traj.model_dump_json(indent=2), encoding="utf-8")

    out_rollouts = tmp_path / "rollouts.json"
    out_index = tmp_path / "index.json"
    rc = builder.main(
        [
            "--trajectory-dir",
            str(in_dir),
            "--precision-class",
            "bf16",
            "--out-rollouts",
            str(out_rollouts),
            "--out-index",
            str(out_index),
        ]
    )
    assert rc == 0
    rollouts = json.loads(out_rollouts.read_text())
    index = json.loads(out_index.read_text())
    # 1 + 2 turns = 3 entries.
    assert len(rollouts) == 3
    assert len(index["entries"]) == 3
    assert index["rollouts_path"] == str(out_rollouts)
    assert index["precision_class"] == "bf16"


def test_cli_aborts_on_missing_directory(builder: ModuleType, tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    rc = builder.main(
        [
            "--trajectory-dir",
            str(missing),
            "--precision-class",
            "bf16",
        ]
    )
    assert rc == 2


def test_cli_aborts_on_empty_directory(builder: ModuleType, tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = builder.main(
        [
            "--trajectory-dir",
            str(empty),
            "--precision-class",
            "bf16",
        ]
    )
    assert rc == 2
