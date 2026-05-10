"""Tests for the agent-trajectory observatory."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
    compute_trajectory_divergence,
    load_trajectory,
    render_trajectory_html,
    write_trajectory_diff,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint


def _fp(name: str) -> EngineFingerprint:
    return EngineFingerprint(name=name, version="0.0", fingerprint=f"sha256:{name}")


def _call(name: str, args: dict | str = "{}", result: str = "ok") -> ToolCallRecord:
    return ToolCallRecord.from_call(name=name, arguments=args, result=result)


def _traj(
    *,
    run_id: str = "r",
    rollout: str = "vllm-bf16",
    steps: list[AgentStep] | None = None,
    final_answer: str = "answer",
    terminal: str = "answered",
    task_id: str = "t-1",
    model_id: str = "Qwen/Qwen3-32B",
) -> AgentTrajectory:
    if steps is None:
        steps = [
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search", {"q": "x"})]),
            AgentStep(step_idx=0, role="tool", content="result-x"),
            AgentStep(step_idx=1, role="assistant", content=final_answer),
        ]
    return AgentTrajectory(
        run_id=run_id,
        task_id=task_id,
        task_text="solve the thing",
        model_id=model_id,
        rollout_engine=_fp(rollout),
        available_tools=["search", "calculator"],
        steps=steps,
        success=True,
        terminal_reason=terminal,
        total_assistant_tokens=10,
        final_answer=final_answer,
    )


# --- contract ---------------------------------------------------------------


def test_tool_call_record_hashes_match_when_args_match():
    a = ToolCallRecord.from_call(name="search", arguments={"q": "x"}, result="r")
    b = ToolCallRecord.from_call(name="search", arguments={"q": "x"}, result="r")
    assert a.arguments_hash == b.arguments_hash
    assert a.result_hash == b.result_hash


def test_tool_call_record_hashes_change_with_args():
    a = ToolCallRecord.from_call(name="search", arguments={"q": "x"}, result="r")
    b = ToolCallRecord.from_call(name="search", arguments={"q": "y"}, result="r")
    assert a.arguments_hash != b.arguments_hash


def test_agent_step_rejects_tool_calls_on_tool_role():
    with pytest.raises(Exception):
        AgentStep(
            step_idx=0,
            role="tool",
            content="r",
            tool_calls=[_call("search")],
        )


def test_agent_trajectory_assistant_steps_filters():
    t = _traj()
    assert len(t.assistant_steps()) == 2
    assert all(s.role == "assistant" for s in t.assistant_steps())


def test_tool_call_signature_includes_only_tool_calls():
    t = _traj()
    sig = t.tool_call_signature()
    assert len(sig) == 1
    assert sig[0][0] == "search"


# --- divergence metric ------------------------------------------------------


def test_identical_trajectories_have_no_divergence():
    a = _traj(run_id="a", rollout="vllm-bf16")
    b = _traj(run_id="b", rollout="vllm-fp8")
    rep = compute_trajectory_divergence(a, b)
    assert rep.first_divergence_step is None
    assert rep.first_divergence_kind is None
    assert rep.tool_choice_disagreement_rate == 0.0
    assert rep.tool_call_jaccard == 1.0
    assert rep.final_answer_match is True


def test_divergence_on_tool_choice():
    base_steps = [
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r1"),
        AgentStep(step_idx=1, role="assistant", tool_calls=[_call("calculator")]),
        AgentStep(step_idx=1, role="tool", content="r2"),
        AgentStep(step_idx=2, role="assistant", content="answer"),
    ]
    a = _traj(steps=base_steps, run_id="a", rollout="vllm-bf16")
    diverged = base_steps[:2] + [
        AgentStep(step_idx=1, role="assistant", tool_calls=[_call("python_eval")]),
        AgentStep(step_idx=1, role="tool", content="r2"),
        AgentStep(step_idx=2, role="assistant", content="answer"),
    ]
    b = _traj(steps=diverged, run_id="b", rollout="vllm-fp8")
    rep = compute_trajectory_divergence(b, a)  # b is rollout, a is trainer
    assert rep.first_divergence_step == 1
    assert rep.first_divergence_kind == "tool_choice"
    assert rep.tool_choice_disagreement_rate > 0.0


def test_divergence_on_tool_args():
    a = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search", {"q": "X"})]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", content="answer"),
    ], rollout="vllm-bf16")
    b = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search", {"q": "Y"})]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", content="answer"),
    ], rollout="vllm-fp8")
    rep = compute_trajectory_divergence(b, a)
    assert rep.first_divergence_step == 0
    assert rep.first_divergence_kind == "tool_args"


def test_divergence_on_termination_when_prefix_matches_but_lengths_differ():
    short = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", content="answer"),
    ], rollout="vllm-bf16", final_answer="answer")
    long = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", content="answer"),
        AgentStep(step_idx=2, role="assistant", content="follow-up clarification"),
    ], rollout="vllm-fp8", final_answer="follow-up clarification")
    rep = compute_trajectory_divergence(long, short)
    assert rep.first_divergence_step == 2  # past the matching prefix
    assert rep.first_divergence_kind == "termination"
    assert rep.same_step_count is False


def test_divergence_when_long_traj_takes_extra_tool_step_is_tool_choice_not_termination():
    short = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", content="answer"),
    ], rollout="vllm-bf16")
    long = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=1, role="tool", content="r2"),
        AgentStep(step_idx=2, role="assistant", content="answer"),
    ], rollout="vllm-fp8")
    # At step 1 the long traj calls a tool while the short one just answers —
    # that's a tool_choice divergence, not a termination one.
    rep = compute_trajectory_divergence(long, short)
    assert rep.first_divergence_step == 1
    assert rep.first_divergence_kind == "tool_choice"
    assert rep.same_step_count is False


def test_divergence_on_content_only():
    a = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", content="answer A"),
    ], rollout="vllm-bf16", final_answer="answer A")
    b = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", content="answer B"),
    ], rollout="vllm-fp8", final_answer="answer B")
    rep = compute_trajectory_divergence(b, a)
    assert rep.first_divergence_step == 0
    assert rep.first_divergence_kind == "content"


def test_jaccard_partial_overlap():
    a = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", tool_calls=[_call("calculator")]),
        AgentStep(step_idx=1, role="tool", content="r2"),
        AgentStep(step_idx=2, role="assistant", content="answer"),
    ])
    b = _traj(steps=[
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", tool_calls=[_call("python_eval")]),
        AgentStep(step_idx=1, role="tool", content="r2"),
        AgentStep(step_idx=2, role="assistant", content="answer"),
    ], rollout="other")
    rep = compute_trajectory_divergence(b, a)
    # Two distinct tools each side, one shared (search) -> jaccard 1/3.
    assert rep.tool_call_jaccard == pytest.approx(1 / 3)


def test_compute_rejects_task_id_mismatch():
    a = _traj(task_id="t1")
    b = _traj(task_id="t2", rollout="x")
    with pytest.raises(ValueError):
        compute_trajectory_divergence(b, a)


def test_compute_rejects_model_id_mismatch():
    a = _traj(model_id="m1")
    b = _traj(model_id="m2", rollout="x")
    with pytest.raises(ValueError):
        compute_trajectory_divergence(b, a)


def test_final_answer_match_when_both_unanswered_and_max_steps():
    a = _traj(final_answer="", terminal="max_steps")
    b = _traj(final_answer="", terminal="max_steps", rollout="x")
    rep = compute_trajectory_divergence(b, a)
    assert rep.final_answer_match is False  # neither answered, not a true match


# --- rendering --------------------------------------------------------------


def test_render_html_self_contained_and_marks_divergence():
    a = _traj(run_id="a")
    b_steps = [
        AgentStep(step_idx=0, role="assistant", tool_calls=[_call("python_eval")]),
        AgentStep(step_idx=0, role="tool", content="r"),
        AgentStep(step_idx=1, role="assistant", content="answer"),
    ]
    b = _traj(run_id="b", steps=b_steps, rollout="vllm-fp8")
    rep = compute_trajectory_divergence(b, a)
    page = render_trajectory_html(rep, b, a)
    assert page.startswith("<!doctype html>")
    assert "first divergence" in page.lower() or "first-div" in page.lower() or "first divergence step" in page.lower()
    assert "side-by-side" in page.lower() or "Side-by-side" in page


def test_render_html_escapes_user_input():
    a = _traj()
    b_steps = [
        AgentStep(step_idx=0, role="assistant", content="<script>alert(1)</script>"),
    ]
    b = _traj(steps=b_steps, rollout="x")
    rep = compute_trajectory_divergence(b, a)
    page = render_trajectory_html(rep, b, a)
    assert "<script>alert" not in page
    assert "&lt;script&gt;" in page


def test_write_trajectory_diff(tmp_path: Path):
    a = _traj()
    b = _traj(rollout="other")
    rep = compute_trajectory_divergence(b, a)
    paths = write_trajectory_diff(rep, b, a, tmp_path / "out")
    assert paths["json"].exists()
    assert paths["html"].exists()
    payload = json.loads(paths["json"].read_text())
    assert payload["schema_version"] == "agent_divergence.v0"


def test_load_trajectory_round_trip(tmp_path: Path):
    t = _traj()
    p = tmp_path / "t.json"
    p.write_text(t.model_dump_json())
    again = load_trajectory(p)
    assert again.run_id == t.run_id
    assert again.steps[0].tool_calls[0].name == "search"
