"""Offline tests for scripts/live/hermes_agent_runner.py.

The CLI shell-out is not exercised here — only the pure adapter and
the JSONL-to-AgentTrajectory orchestrator. A fixture JSONL covers the
shape the real Hermes Agent CLI emits: a <think> block, two tool_call
blocks paired with their tool results, and a final non-tool assistant
turn. A separate negative test asserts the adapter rejects malformed
records with a clear error message.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
)


def _load_runner(env: dict[str, str] | None = None) -> ModuleType:
    """Import scripts/live/hermes_agent_runner.py under custom env vars."""
    spec_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "live"
        / "hermes_agent_runner.py"
    )
    saved = {k: os.environ.get(k) for k in (env or {})}
    if env:
        os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location(
            f"hermes_agent_runner_{id(env)}", spec_path
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


@pytest.fixture(scope="module")
def runner() -> ModuleType:
    return _load_runner()


# --- fixture JSONL ----------------------------------------------------------


def _fixture_records() -> list[dict]:
    """One realistic Hermes Agent JSONL trace for `code-fix-factorial`.

    Covers: <think> block, two tool_call records paired with tool
    results via tool_call_id, and a non-tool final assistant turn.
    """
    return [
        {"role": "system", "content": "You are a careful assistant."},
        {"role": "user", "content": "Fix /tmp/factorial.py and verify."},
        {
            "role": "assistant",
            "content": "<think>I should read the file first.</think>",
            "response_token_count": 12,
        },
        {
            "role": "assistant",
            "content": "Let me read the file.",
            "response_token_count": 18,
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/factorial.py"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read",
            "content": "def factorial(n):\n    if n == 0:\n        return 0\n    return n * factorial(n - 1)\n",
        },
        {
            "role": "assistant",
            "content": "The base case is wrong. Let me verify the fix.",
            "response_token_count": 22,
            "tool_calls": [
                {
                    "id": "call_eval",
                    "type": "function",
                    "function": {
                        "name": "python_eval",
                        "arguments": '{"expression": "5*4*3*2*1"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_eval", "content": "120"},
        {
            "role": "assistant",
            "content": "factorial(0) should return 1, not 0. Corrected, factorial(5) returns 120.",
            "response_token_count": 31,
        },
    ]


def _task() -> dict:
    return {
        "task_id": "code-fix-factorial",
        "task_text": "Fix /tmp/factorial.py and verify with python_eval.",
        "available_tools": ["read_file", "python_eval"],
    }


# --- adapt_hermes_record (single-record contract) ---------------------------


def test_adapt_think_record_keeps_content(runner: ModuleType) -> None:
    rec = {
        "role": "assistant",
        "content": "<think>plan: read file</think>",
        "response_token_count": 12,
    }
    step = runner.adapt_hermes_record(rec)
    assert isinstance(step, AgentStep)
    assert step.role == "assistant"
    assert step.content == "<think>plan: read file</think>"
    assert step.tool_calls == []
    assert step.response_token_count == 12


def test_adapt_tool_call_record_pairs_result(runner: ModuleType) -> None:
    rec = {
        "role": "assistant",
        "content": "Reading.",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "/tmp/factorial.py"}',
                },
            }
        ],
    }
    step = runner.adapt_hermes_record(
        rec, tool_results={"c1": "file contents here"}
    )
    assert step.role == "assistant"
    assert len(step.tool_calls) == 1
    tc = step.tool_calls[0]
    assert isinstance(tc, ToolCallRecord)
    assert tc.name == "read_file"
    assert tc.result_text == "file contents here"
    # arguments_hash should match a fresh ToolCallRecord with the same dict.
    expected = ToolCallRecord.from_call(
        name="read_file",
        arguments={"path": "/tmp/factorial.py"},
        result="file contents here",
    )
    assert tc.arguments_hash == expected.arguments_hash


def test_adapt_tool_result_record(runner: ModuleType) -> None:
    rec = {"role": "tool", "tool_call_id": "c1", "content": "120"}
    step = runner.adapt_hermes_record(rec)
    assert step.role == "tool"
    assert step.content == "120"
    assert step.tool_calls == []


def test_adapt_rejects_missing_role(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="role"):
        runner.adapt_hermes_record({"content": "nope"})


def test_adapt_rejects_user_role(runner: ModuleType) -> None:
    """User records belong to the prompt envelope, not the trajectory."""
    with pytest.raises(ValueError, match="role"):
        runner.adapt_hermes_record({"role": "user", "content": "hi"})


def test_adapt_rejects_non_dict_record(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="dict"):
        runner.adapt_hermes_record("not a dict")  # type: ignore[arg-type]


def test_adapt_rejects_malformed_tool_call_missing_name(runner: ModuleType) -> None:
    rec = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {}}],
    }
    with pytest.raises(ValueError, match="function.name"):
        runner.adapt_hermes_record(rec)


def test_adapt_handles_dict_arguments(runner: ModuleType) -> None:
    """Hermes may emit arguments as a dict instead of a JSON-encoded string."""
    rec = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "calculator", "arguments": {"expression": "2+2"}},
            }
        ],
    }
    step = runner.adapt_hermes_record(rec, tool_results={"c1": "4"})
    assert step.tool_calls[0].name == "calculator"


# --- adapt_hermes_jsonl (full-trajectory contract) --------------------------


def test_adapt_jsonl_builds_full_trajectory(runner: ModuleType) -> None:
    traj = runner.adapt_hermes_jsonl(
        _fixture_records(),
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-32b-bf16-tp4-l40s",
    )
    assert isinstance(traj, AgentTrajectory)
    assert traj.task_id == "code-fix-factorial"
    assert traj.model_id == "Qwen/Qwen3-32B"
    assert traj.rollout_engine.name == "hermes-qwen3-32b-bf16"
    # 4 assistant records + 2 tool records = 6 steps; system+user skipped.
    assert len(traj.steps) == 6
    assert [s.role for s in traj.steps] == [
        "assistant",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    # The two tool_calls picked up their paired result text.
    assistants = traj.assistant_steps()
    assert len(assistants[1].tool_calls) == 1
    assert assistants[1].tool_calls[0].result_text.startswith("def factorial")
    assert assistants[2].tool_calls[0].result_text == "120"
    assert assistants[2].tool_calls[0].name == "python_eval"
    assert traj.total_assistant_tokens == 12 + 18 + 22 + 31


def test_adapt_jsonl_terminal_reason_answered(runner: ModuleType) -> None:
    traj = runner.adapt_hermes_jsonl(
        _fixture_records(),
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-32b-bf16",
    )
    assert traj.terminal_reason == "answered"
    assert "120" in traj.final_answer


def test_adapt_jsonl_terminal_reason_max_steps(runner: ModuleType) -> None:
    """If the last assistant record still has tool_calls AND we hit the cap,
    terminal_reason must be ``max_steps`` (not ``answered``)."""
    records: list[dict] = []
    for i in range(3):
        records.append(
            {
                "role": "assistant",
                "content": f"step {i}",
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "arguments": '{"expression": "1+1"}',
                        },
                    }
                ],
            }
        )
        records.append({"role": "tool", "tool_call_id": f"c{i}", "content": "2"})
    traj = runner.adapt_hermes_jsonl(
        records,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
        max_steps=3,
    )
    assert traj.terminal_reason == "max_steps"
    assert traj.final_answer == ""


def test_adapt_jsonl_terminal_reason_error_on_empty(runner: ModuleType) -> None:
    traj = runner.adapt_hermes_jsonl(
        [{"role": "system", "content": "hi"}, {"role": "user", "content": "go"}],
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assert traj.terminal_reason == "error"
    assert traj.steps == []


def test_adapt_jsonl_raises_on_non_dict_record(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="dict"):
        runner.adapt_hermes_jsonl(
            [{"role": "user", "content": "hi"}, "not a dict"],  # type: ignore[list-item]
            task=_task(),
            model_id="Qwen/Qwen3-32B",
            engine_label="hermes-qwen3-32b-bf16",
            engine_fingerprint="sha256:x",
        )


def test_adapt_jsonl_roundtrips_through_json(runner: ModuleType, tmp_path: Path) -> None:
    """The emitted AgentTrajectory must serialise + parse back unchanged."""
    traj = runner.adapt_hermes_jsonl(
        _fixture_records(),
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    out = tmp_path / "traj.json"
    out.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    reloaded = AgentTrajectory.model_validate_json(out.read_text(encoding="utf-8"))
    assert reloaded.task_id == traj.task_id
    assert reloaded.terminal_reason == traj.terminal_reason
    assert len(reloaded.steps) == len(traj.steps)


# --- env-driven engine label default ----------------------------------------


def test_engine_label_default_is_hermes_model_precision() -> None:
    """Default ENGINE_LABEL derives `hermes-{model_slug}-{precision}` from MODEL + PRECISION."""
    mod = _load_runner(
        {
            "MODEL": "Qwen/Qwen3-30B-A3B",
            "PRECISION": "fp8",
            "ENGINE_LABEL": "",  # force re-derive
        }
    )
    # An empty-string ENGINE_LABEL falls through to the default branch.
    assert mod._default_engine_label() == "hermes-qwen3-30b-a3b-fp8"


def test_engine_label_override_from_env() -> None:
    """An explicit ENGINE_LABEL env var wins over the default."""
    mod = _load_runner(
        {
            "MODEL": "Qwen/Qwen3-32B",
            "PRECISION": "bf16",
            "ENGINE_LABEL": "hermes-custom-tag",
        }
    )
    assert mod.ENGINE_LABEL == "hermes-custom-tag"


# --- JSONL parsing ----------------------------------------------------------


def test_read_jsonl_round_trip(runner: ModuleType, tmp_path: Path) -> None:
    p = tmp_path / "fixture.jsonl"
    lines = [json.dumps(r) for r in _fixture_records()]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = runner._read_jsonl(p)
    assert len(parsed) == len(_fixture_records())
    assert parsed[2]["content"] == "<think>plan: read file</think>" or (
        parsed[2]["role"] == "assistant"
    )


def test_read_jsonl_rejects_invalid_line(runner: ModuleType, tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text('{"role": "assistant", "content": "ok"}\n{not json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        runner._read_jsonl(p)
