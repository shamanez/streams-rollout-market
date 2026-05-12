"""Offline tests for scripts/live/fill_prompt_token_ids.py.

Uses the same FakeTokenizer pattern as test_hermes_token_extraction.py
so the test layer doesn't pull HF weights. The filler is a thin
wrapper over hermes_token_extraction's `_build_prior_messages` +
`_coerce_to_id_list`; the tests focus on the integration:

  * a single proxy-captured turn (response IDs present, prompt
    missing) gets its prompt re-derived from the trajectory history.
  * fully captured turns are left untouched.
  * fully legacy turns (no response IDs either) are NOT auto-filled
    — the filler only handles the partial-capture path.
  * dry-run leaves disk untouched.
  * CLI round-trip writes a probe-bearing trajectory back to the same
    file with prompt_token_ids set.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint


def _load_filler() -> ModuleType:
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "fill_prompt_token_ids.py"
    )
    spec = importlib.util.spec_from_file_location("fill_prompt_token_ids", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def filler() -> ModuleType:
    return _load_filler()


class FakeTokenizer:
    """Tiny chat-template stub: returns a deterministic list of ints
    keyed by role + content. Same shape used by the extractor tests."""

    ROLE_OPEN: dict[str, int] = {
        "system": 90001,
        "user": 90002,
        "assistant": 90003,
        "tool": 90004,
    }
    ROLE_CLOSE: int = 90099
    GEN_PROMPT_MARKER: int = 90100

    def apply_chat_template(
        self,
        conversation: list[dict],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
    ) -> list[int]:
        assert tokenize is True
        ids: list[int] = []
        for msg in conversation:
            role = msg["role"]
            ids.append(self.ROLE_OPEN[role])
            if role == "assistant":
                ids.append(self.GEN_PROMPT_MARKER)
            ids.extend(ord(c) for c in msg["content"])
            ids.append(self.ROLE_CLOSE)
        if add_generation_prompt:
            ids.append(self.ROLE_OPEN["assistant"])
            ids.append(self.GEN_PROMPT_MARKER)
        return ids


def _engine() -> EngineFingerprint:
    return EngineFingerprint(
        name="hermes-qwen3-32b-bf16",
        version="0.20.1",
        fingerprint="sha256:hermes-qwen3-32b-bf16",
    )


def _traj_with_partial_capture(*, fill_prompt_on: list[int] | None = None) -> AgentTrajectory:
    """Build a two-assistant-turn trajectory; both turns have
    response_token_ids but neither has prompt_token_ids unless
    ``fill_prompt_on`` mentions their assistant-idx."""
    fill_prompt_on = fill_prompt_on or []
    steps: list[AgentStep] = []
    for i in range(2):
        kwargs = {
            "step_idx": i,
            "role": "assistant",
            "content": f"turn {i}",
            "response_token_ids": [100 + i * 10, 101 + i * 10, 102 + i * 10],
            "response_logprobs": [-0.1, -0.2, -0.3],
        }
        if i in fill_prompt_on:
            kwargs["prompt_token_ids"] = [9, 9, 9]
        steps.append(AgentStep(**kwargs))
    return AgentTrajectory(
        run_id="hermes-qwen3-32b-bf16-t1",
        task_id="t1",
        task_text="what is 2+2?",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_engine(),
        available_tools=[],
        steps=steps,
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
    )


def test_fill_partial_capture_turn(filler: ModuleType) -> None:
    """A turn with response_token_ids but no prompt_token_ids gets
    its prompt re-derived via chat-template encoding."""
    traj = _traj_with_partial_capture()
    out, n = filler.fill_trajectory(traj, FakeTokenizer())
    assert n == 2
    assistants = [s for s in out.steps if s.role == "assistant"]
    for step in assistants:
        assert step.prompt_token_ids is not None
        # Chat-template fallback ends at the assistant generation marker.
        assert step.prompt_token_ids[-1] == FakeTokenizer.GEN_PROMPT_MARKER
        # Captured response IDs were preserved verbatim.
        assert step.response_token_ids is not None


def test_fill_leaves_already_populated_turns(filler: ModuleType) -> None:
    """A turn that already has prompt_token_ids set passes through
    unchanged; only None-prompt turns get filled."""
    traj = _traj_with_partial_capture(fill_prompt_on=[0])
    out, n = filler.fill_trajectory(traj, FakeTokenizer())
    assert n == 1  # only turn 1 needed filling
    assistants = [s for s in out.steps if s.role == "assistant"]
    # Turn 0 still has its sentinel prompt [9, 9, 9] — unmodified.
    assert assistants[0].prompt_token_ids == [9, 9, 9]
    # Turn 1 got a re-derived chat-template prompt.
    assert assistants[1].prompt_token_ids != [9, 9, 9]
    assert assistants[1].prompt_token_ids is not None
    assert assistants[1].prompt_token_ids[-1] == FakeTokenizer.GEN_PROMPT_MARKER


def test_fill_skips_legacy_turns_without_response_ids(filler: ModuleType) -> None:
    """The filler only handles the partial-capture path; fully legacy
    turns (no response_token_ids) are left alone so the existing
    extract_token_pairs fallback can handle them downstream."""
    traj = AgentTrajectory(
        run_id="r1",
        task_id="t1",
        task_text="x",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_engine(),
        available_tools=[],
        steps=[
            AgentStep(step_idx=0, role="assistant", content="answer"),
        ],
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
    )
    out, n = filler.fill_trajectory(traj, FakeTokenizer())
    assert n == 0
    assert out is traj  # same instance: no copy needed


def test_fill_history_includes_tool_steps(filler: ModuleType) -> None:
    """The re-derived prompt for turn k must include every prior
    tool step (vLLM saw them as part of the accumulated history)."""
    traj = AgentTrajectory(
        run_id="r1",
        task_id="t1",
        task_text="multi-turn",
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_engine(),
        available_tools=[],
        steps=[
            AgentStep(
                step_idx=0,
                role="assistant",
                content="t0",
                response_token_ids=[1, 2, 3],
                response_logprobs=[-0.1, -0.2, -0.3],
            ),
            AgentStep(step_idx=0, role="tool", content="result"),
            AgentStep(
                step_idx=1,
                role="assistant",
                content="t1",
                response_token_ids=[4, 5, 6],
                response_logprobs=[-0.1, -0.2, -0.3],
            ),
        ],
        success=None,
        terminal_reason="answered",
        total_assistant_tokens=0,
        final_answer="",
    )
    out, n = filler.fill_trajectory(traj, FakeTokenizer())
    assert n == 2
    assistants = [s for s in out.steps if s.role == "assistant"]
    p1 = assistants[1].prompt_token_ids
    assert p1 is not None
    # Turn-1 prompt must include the tool role-open from the
    # intervening tool step.
    assert FakeTokenizer.ROLE_OPEN["tool"] in p1
    # Turn-0 prompt did NOT see the tool yet.
    p0 = assistants[0].prompt_token_ids
    assert p0 is not None
    assert FakeTokenizer.ROLE_OPEN["tool"] not in p0


def test_cli_writes_filled_trajectory_back(
    filler: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: CLI walks dir, fills missing prompt IDs, writes back."""
    monkeypatch.setattr(filler, "_load_tokenizer", lambda model_id: FakeTokenizer())
    traj = _traj_with_partial_capture()
    in_dir = tmp_path / "traj"
    in_dir.mkdir()
    out_path = in_dir / "t1.json"
    out_path.write_text(traj.model_dump_json(indent=2), encoding="utf-8")

    rc = filler.main(["--trajectory-dir", str(in_dir)])
    assert rc == 0
    from rollout_market.observatory.agent_trajectory_lab import load_trajectory

    reloaded = load_trajectory(out_path)
    assistants = [s for s in reloaded.steps if s.role == "assistant"]
    assert all(s.prompt_token_ids is not None for s in assistants)


def test_cli_dry_run_does_not_write(
    filler: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(filler, "_load_tokenizer", lambda model_id: FakeTokenizer())
    traj = _traj_with_partial_capture()
    in_dir = tmp_path / "traj"
    in_dir.mkdir()
    out_path = in_dir / "t1.json"
    original = traj.model_dump_json(indent=2)
    out_path.write_text(original, encoding="utf-8")

    rc = filler.main(["--trajectory-dir", str(in_dir), "--dry-run"])
    assert rc == 0
    # Disk content is byte-identical to the pre-run snapshot.
    assert out_path.read_text(encoding="utf-8") == original
