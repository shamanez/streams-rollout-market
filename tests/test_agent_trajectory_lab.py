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


# --- rollout-time probe fields (cycle-3) -----------------------------------


def test_agent_step_accepts_rollout_probes_on_assistant():
    step = AgentStep(
        step_idx=0,
        role="assistant",
        content="hi",
        prompt_token_ids=[1, 2, 3, 4],
        response_token_ids=[10, 20, 30],
        response_logprobs=[-0.1, -0.2, -0.3],
        response_routed_experts=[
            [[0, 1], [2, 3]],
            [[4, 5], [6, 7]],
            [[8, 9], [10, 11]],
        ],
    )
    assert step.prompt_token_ids == [1, 2, 3, 4]
    assert step.response_token_ids == [10, 20, 30]
    assert step.response_logprobs == [-0.1, -0.2, -0.3]
    assert len(step.response_routed_experts) == 3


def test_agent_step_defaults_probes_to_none():
    step = AgentStep(step_idx=0, role="assistant", content="hi")
    assert step.prompt_token_ids is None
    assert step.response_token_ids is None
    assert step.response_logprobs is None
    assert step.response_routed_experts is None


def test_agent_step_rejects_logprob_length_mismatch():
    with pytest.raises(Exception, match="response_logprobs length"):
        AgentStep(
            step_idx=0,
            role="assistant",
            content="hi",
            response_token_ids=[10, 20, 30],
            response_logprobs=[-0.1, -0.2],
        )


def test_agent_step_rejects_routed_experts_length_mismatch():
    with pytest.raises(Exception, match="response_routed_experts length"):
        AgentStep(
            step_idx=0,
            role="assistant",
            content="hi",
            response_token_ids=[10, 20],
            response_routed_experts=[[[0]], [[1]], [[2]]],
        )


def test_agent_step_rejects_probes_on_tool_role():
    with pytest.raises(Exception, match="tool-role steps must not carry"):
        AgentStep(
            step_idx=0,
            role="tool",
            content="result",
            response_token_ids=[10, 20],
        )
    with pytest.raises(Exception, match="tool-role steps must not carry"):
        AgentStep(
            step_idx=0,
            role="tool",
            content="result",
            response_logprobs=[-0.1, -0.2],
        )


def test_agent_step_round_trips_probes_through_json():
    step = AgentStep(
        step_idx=0,
        role="assistant",
        content="hi",
        prompt_token_ids=[1, 2, 3],
        response_token_ids=[10, 20],
        response_logprobs=[-0.1, -0.2],
        response_routed_experts=[[[0, 1]], [[2, 3]]],
    )
    js = step.model_dump_json()
    revived = AgentStep.model_validate_json(js)
    assert revived.prompt_token_ids == [1, 2, 3]
    assert revived.response_token_ids == [10, 20]
    assert revived.response_logprobs == [-0.1, -0.2]
    assert revived.response_routed_experts == [[[0, 1]], [[2, 3]]]


def test_agent_step_loads_legacy_json_without_probe_fields():
    """A trajectory captured before the probes existed must validate.

    The on-disk schema for cycle-2 hermes-agent rollouts had no
    prompt_token_ids / response_token_ids / response_logprobs /
    response_routed_experts keys at all. The new fields default to
    None so model_validate_json on those legacy payloads still passes.
    """
    legacy_json = (
        '{"step_idx": 0, "role": "assistant", "content": "hi", '
        '"tool_calls": [], "finish_reason": null, '
        '"response_token_count": 0, "response_logprobs_hash": null}'
    )
    revived = AgentStep.model_validate_json(legacy_json)
    assert revived.role == "assistant"
    assert revived.prompt_token_ids is None
    assert revived.response_logprobs is None


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
    assert rep.final_answer_match_soft is True


# --- soft answer match -----------------------------------------------------


def test_soft_match_picks_up_same_number_different_phrasing():
    """Surface phrasing changes around the same number must soft-match."""
    a = _traj(
        run_id="a", rollout="vllm-bf16", final_answer="The total is **8,294,400** tokens per day."
    )
    b = _traj(run_id="b", rollout="vllm-fp8", final_answer="That comes to 8294400 tokens daily.")
    rep = compute_trajectory_divergence(a, b)
    assert rep.final_answer_match is False  # strict normalised-text equality fails
    assert rep.final_answer_match_soft is True  # numeric equivalence rescues it


def test_soft_match_rejects_different_numbers():
    a = _traj(run_id="a", rollout="vllm-bf16", final_answer="The total is 8294400 tokens.")
    b = _traj(run_id="b", rollout="vllm-fp8", final_answer="The total is 8295000 tokens.")
    rep = compute_trajectory_divergence(a, b)
    assert rep.final_answer_match is False
    assert rep.final_answer_match_soft is False


def test_soft_match_picks_up_substring():
    """The shorter answer is contained in the longer one."""
    a = _traj(run_id="a", rollout="vllm-bf16", final_answer="The answer is 42.")
    b = _traj(
        run_id="b", rollout="vllm-fp8", final_answer="After careful analysis, the answer is 42."
    )
    rep = compute_trajectory_divergence(a, b)
    assert rep.final_answer_match is False
    assert rep.final_answer_match_soft is True


def test_soft_match_strict_equality_still_implies_soft():
    a = _traj(run_id="a", rollout="vllm-bf16", final_answer="same answer")
    b = _traj(run_id="b", rollout="vllm-fp8", final_answer="Same Answer")
    rep = compute_trajectory_divergence(a, b)
    assert rep.final_answer_match is True
    assert rep.final_answer_match_soft is True


def test_soft_match_handles_decimal_normalisation():
    """1.20 and 1.2 are the same number after canonicalisation."""
    from rollout_market.observatory.agent_trajectory_lab import _soft_content_match

    assert _soft_content_match("The ratio is 1.20.", "The ratio is 1.2.")


def test_soft_match_handles_thousands_separator():
    from rollout_market.observatory.agent_trajectory_lab import _soft_content_match

    assert _soft_content_match("1,200,000 tokens", "1200000 tokens")
    assert _soft_content_match("1,200,000", "1200000")


def test_soft_match_returns_false_when_one_side_empty():
    a = _traj(run_id="a", rollout="vllm-bf16", final_answer="The answer is 5.", terminal="answered")
    # one side empty + the other has content -> no soft match
    b = _traj(
        run_id="b",
        rollout="vllm-fp8",
        final_answer="",
        terminal="max_steps",
    )
    rep = compute_trajectory_divergence(b, a)
    assert rep.final_answer_match is False
    assert rep.final_answer_match_soft is False


def test_soft_match_both_empty_with_answered_terminal_is_true():
    """The existing empty-empty rule (both answered, both empty answer) keeps
    its strict behaviour and propagates to the soft flag."""
    a = _traj(run_id="a", rollout="vllm-bf16", final_answer="", terminal="answered")
    b = _traj(run_id="b", rollout="vllm-fp8", final_answer="", terminal="answered")
    rep = compute_trajectory_divergence(a, b)
    assert rep.final_answer_match is True
    assert rep.final_answer_match_soft is True


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
    a = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search", {"q": "X"})]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", content="answer"),
        ],
        rollout="vllm-bf16",
    )
    b = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search", {"q": "Y"})]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", content="answer"),
        ],
        rollout="vllm-fp8",
    )
    rep = compute_trajectory_divergence(b, a)
    assert rep.first_divergence_step == 0
    assert rep.first_divergence_kind == "tool_args"


def test_divergence_on_termination_when_prefix_matches_but_lengths_differ():
    short = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", content="answer"),
        ],
        rollout="vllm-bf16",
        final_answer="answer",
    )
    long = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", content="answer"),
            AgentStep(step_idx=2, role="assistant", content="follow-up clarification"),
        ],
        rollout="vllm-fp8",
        final_answer="follow-up clarification",
    )
    rep = compute_trajectory_divergence(long, short)
    assert rep.first_divergence_step == 2  # past the matching prefix
    assert rep.first_divergence_kind == "termination"
    assert rep.same_step_count is False


def test_divergence_when_long_traj_takes_extra_tool_step_is_tool_choice_not_termination():
    short = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", content="answer"),
        ],
        rollout="vllm-bf16",
    )
    long = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=1, role="tool", content="r2"),
            AgentStep(step_idx=2, role="assistant", content="answer"),
        ],
        rollout="vllm-fp8",
    )
    # At step 1 the long traj calls a tool while the short one just answers —
    # that's a tool_choice divergence, not a termination one.
    rep = compute_trajectory_divergence(long, short)
    assert rep.first_divergence_step == 1
    assert rep.first_divergence_kind == "tool_choice"
    assert rep.same_step_count is False


def test_divergence_on_content_only():
    a = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", content="answer A"),
        ],
        rollout="vllm-bf16",
        final_answer="answer A",
    )
    b = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", content="answer B"),
        ],
        rollout="vllm-fp8",
        final_answer="answer B",
    )
    rep = compute_trajectory_divergence(b, a)
    assert rep.first_divergence_step == 0
    assert rep.first_divergence_kind == "content"


def test_jaccard_partial_overlap():
    a = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", tool_calls=[_call("calculator")]),
            AgentStep(step_idx=1, role="tool", content="r2"),
            AgentStep(step_idx=2, role="assistant", content="answer"),
        ]
    )
    b = _traj(
        steps=[
            AgentStep(step_idx=0, role="assistant", tool_calls=[_call("search")]),
            AgentStep(step_idx=0, role="tool", content="r"),
            AgentStep(step_idx=1, role="assistant", tool_calls=[_call("python_eval")]),
            AgentStep(step_idx=1, role="tool", content="r2"),
            AgentStep(step_idx=2, role="assistant", content="answer"),
        ],
        rollout="other",
    )
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
    assert (
        "first divergence" in page.lower()
        or "first-div" in page.lower()
        or "first divergence step" in page.lower()
    )
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
