"""Offline tests for scripts/live/merge_probes_into_trajectories.py.

The merger stitches a hermes-agent ShareGPT trajectory JSONL together
with a vLLM-side probe sidecar JSONL (keyed by ``prompt_index`` +
``turn_idx``) and adapts each to a probe-bearing ``AgentTrajectory``.

Tests cover:

  * happy path — every assistant turn gets its captured probes; the
    output AgentStep carries them.
  * partial probes — a turn missing a probe entry is tolerated; the
    step has ``response_logprobs is None``.
  * duplicate probe keys raise ``ValueError``.
  * length mismatch between sharegpt records and tasks raises.
  * CLI round-trip writes per-task JSONs that re-parse with probes.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from rollout_market.observatory.agent_trajectory_lab import AgentTrajectory


def _load_module() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "live"
        / "merge_probes_into_trajectories.py"
    )
    spec = importlib.util.spec_from_file_location("merge_probes_into_trajectories", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def merger() -> ModuleType:
    return _load_module()


def _sharegpt_two_turn_record(prompt_index: int = 0) -> dict:
    """One ShareGPT record with two from=gpt turns separated by a tool call."""
    return {
        "prompt_index": prompt_index,
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "q"},
            {
                "from": "gpt",
                "value": (
                    "<tool_call>\n"
                    '{"name": "calculator", "arguments": {"expression": "2+2"}}\n'
                    "</tool_call>"
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "c1", "name": "calculator", "content": "4"}\n'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "<think>\n</think>\nAnswer: 4"},
        ],
        "completed": True,
        "partial": False,
    }


def _tasks_two() -> list[dict]:
    return [
        {"task_id": "math-q1", "task_text": "q1", "available_tools": ["calculator"]},
        {"task_id": "math-q2", "task_text": "q2", "available_tools": ["calculator"]},
    ]


def _probe(prompt_index: int, turn_idx: int, *, n: int = 3) -> dict:
    """Build a probe record. ``n`` = number of response tokens."""
    return {
        "prompt_index": prompt_index,
        "turn_idx": turn_idx,
        "prompt_token_ids": list(range(10, 10 + 4 * (turn_idx + 1))),
        "response_token_ids": [100 + turn_idx * 10 + i for i in range(n)],
        "response_logprobs": [-0.1 - turn_idx * 0.01 - i * 0.001 for i in range(n)],
    }


def test_index_probes_groups_by_key(merger: ModuleType) -> None:
    probes = [
        _probe(0, 0),
        _probe(0, 1),
        _probe(1, 0),
    ]
    idx = merger.index_probes(probes)
    assert set(idx.keys()) == {(0, 0), (0, 1), (1, 0)}
    # Extraneous fields would be dropped — here only load-bearing
    # keys remain on each value.
    assert set(idx[(0, 0)].keys()) <= {
        "prompt_token_ids",
        "response_token_ids",
        "response_logprobs",
        "response_routed_experts",
    }


def test_index_probes_rejects_duplicates(merger: ModuleType) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        merger.index_probes([_probe(0, 0), _probe(0, 0)])


def test_index_probes_rejects_missing_keys(merger: ModuleType) -> None:
    with pytest.raises(ValueError, match="prompt_index"):
        merger.index_probes([{"turn_idx": 0}])


def test_inject_probes_stamps_onto_gpt_entries(merger: ModuleType) -> None:
    sg = _sharegpt_two_turn_record()
    probes_by_turn = {
        0: {
            "prompt_token_ids": [1, 2, 3],
            "response_token_ids": [10, 11, 12],
            "response_logprobs": [-0.1, -0.2, -0.3],
        },
        1: {
            "prompt_token_ids": [1, 2, 3, 4, 5],
            "response_token_ids": [20, 21],
            "response_logprobs": [-0.05, -0.1],
        },
    }
    out = merger.inject_probes(sg, probes_by_turn)
    # Only the two from=gpt entries were modified.
    gpts = [c for c in out["conversations"] if c.get("from") == "gpt"]
    assert len(gpts) == 2
    assert gpts[0]["prompt_token_ids"] == [1, 2, 3]
    assert gpts[0]["response_logprobs"] == [-0.1, -0.2, -0.3]
    assert gpts[1]["response_token_ids"] == [20, 21]
    # system/human/tool entries left untouched.
    for c in out["conversations"]:
        if c.get("from") != "gpt":
            assert "prompt_token_ids" not in c


def test_merge_to_trajectories_threads_probes_through(merger: ModuleType) -> None:
    sg_records = [_sharegpt_two_turn_record(0), _sharegpt_two_turn_record(1)]
    probes = [
        _probe(0, 0),
        _probe(0, 1),
        _probe(1, 0),
        _probe(1, 1),
    ]
    trajs = merger.merge_to_trajectories(
        sharegpt_records=sg_records,
        probe_records=probes,
        tasks=_tasks_two(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-32b-bf16",
    )
    assert len(trajs) == 2
    for k, traj in enumerate(trajs):
        assistants = [s for s in traj.steps if s.role == "assistant"]
        assert len(assistants) == 2
        for turn_idx, step in enumerate(assistants):
            assert step.response_logprobs is not None, f"task {k} turn {turn_idx} missing logprobs"
            assert step.response_token_ids is not None
            assert step.prompt_token_ids is not None


def test_merge_tolerates_missing_probes_on_some_turns(merger: ModuleType) -> None:
    """A trajectory with sparse probe coverage still produces a valid
    AgentTrajectory; only the probe-covered turns carry probes."""
    sg = _sharegpt_two_turn_record(0)
    # Only turn 0 has a probe entry; turn 1 is bare.
    probes = [_probe(0, 0)]
    trajs = merger.merge_to_trajectories(
        sharegpt_records=[sg],
        probe_records=probes,
        tasks=[_tasks_two()[0]],
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assistants = [s for s in trajs[0].steps if s.role == "assistant"]
    assert assistants[0].response_logprobs is not None
    assert assistants[1].response_logprobs is None
    assert assistants[1].prompt_token_ids is None


def test_merge_rejects_length_mismatch(merger: ModuleType) -> None:
    with pytest.raises(ValueError, match="align 1:1"):
        merger.merge_to_trajectories(
            sharegpt_records=[_sharegpt_two_turn_record(0)],
            probe_records=[],
            tasks=_tasks_two(),
            model_id="Qwen/Qwen3-32B",
            engine_label="hermes-qwen3-32b-bf16",
            engine_fingerprint="sha256:x",
        )


def test_cli_writes_probe_bearing_trajectories(merger: ModuleType, tmp_path: Path) -> None:
    sg_records = [_sharegpt_two_turn_record(0), _sharegpt_two_turn_record(1)]
    probes = [_probe(0, 0), _probe(0, 1), _probe(1, 0), _probe(1, 1)]
    sg_path = tmp_path / "sharegpt.jsonl"
    probes_path = tmp_path / "probes.jsonl"
    tasks_path = tmp_path / "tasks.json"
    out_dir = tmp_path / "out"
    sg_path.write_text("\n".join(json.dumps(r) for r in sg_records) + "\n", encoding="utf-8")
    probes_path.write_text("\n".join(json.dumps(r) for r in probes) + "\n", encoding="utf-8")
    tasks_path.write_text(json.dumps(_tasks_two()), encoding="utf-8")

    rc = merger.main(
        [
            "--sharegpt",
            str(sg_path),
            "--probes",
            str(probes_path),
            "--tasks",
            str(tasks_path),
            "--model-id",
            "Qwen/Qwen3-32B",
            "--engine-label",
            "hermes-qwen3-32b-bf16",
            "--engine-fingerprint",
            "sha256:hermes-qwen3-32b-bf16-tp4-l40s",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    files = sorted(out_dir.glob("*.json"))
    assert [p.name for p in files] == ["math-q1.json", "math-q2.json"]
    parsed = AgentTrajectory.model_validate_json(files[0].read_text())
    assistants = [s for s in parsed.steps if s.role == "assistant"]
    assert assistants[0].response_logprobs is not None
    assert assistants[0].prompt_token_ids is not None
