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
    spec_path = Path(__file__).resolve().parents[1] / "scripts" / "live" / "hermes_agent_runner.py"
    saved = {k: os.environ.get(k) for k in (env or {})}
    if env:
        os.environ.update(env)
    try:
        spec = importlib.util.spec_from_file_location(f"hermes_agent_runner_{id(env)}", spec_path)
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
    step = runner.adapt_hermes_record(rec, tool_results={"c1": "file contents here"})
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


# ============================================================================
# ShareGPT adapter (real hermes-agent batch_runner.py output)
# ============================================================================

# This fixture mirrors the exact ShareGPT shape produced by
# hermes-agent/batch_runner.py::process_prompt — one record per prompt with
# the conversation history nested under `conversations`, assistant turns
# embedding `<think>` + `<tool_call>` XML, and tool turns embedding
# `<tool_response>` XML. See hermes-agent/run_agent.py at
# `_convert_to_trajectory_format` for the format spec.


def _sharegpt_record_with_two_tool_calls() -> dict:
    """Realistic Hermes batch_runner output for `code-fix-factorial`."""
    return {
        "prompt_index": 0,
        "conversations": [
            {"from": "system", "value": "You are a function calling AI model. <tools>..."},
            {"from": "human", "value": "Fix /tmp/factorial.py and verify."},
            {
                "from": "gpt",
                "value": (
                    "<think>\nI should read the file first.\n</think>\n"
                    "Let me read the file.\n"
                    "<tool_call>\n"
                    '{"name": "read_file", "arguments": {"path": "/tmp/factorial.py"}}\n'
                    "</tool_call>"
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "call_read", "name": "read_file", "content": "def factorial(n):\\n    if n == 0:\\n        return 0\\n    return n * factorial(n - 1)"}\n'
                    "</tool_response>"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "<think>\nBase case is wrong; let me verify the fix.\n</think>\n"
                    "<tool_call>\n"
                    '{"name": "python_eval", "arguments": {"expression": "5*4*3*2*1"}}\n'
                    "</tool_call>"
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "call_eval", "name": "python_eval", "content": "120"}\n'
                    "</tool_response>"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "<think>\nfactorial(5) returns 120, confirming the corrected base case.\n</think>\n"
                    "The bug: factorial(0) should return 1, not 0. After fix, factorial(5) == 120."
                ),
            },
        ],
        "metadata": {"batch_num": 0, "timestamp": "2026-05-12T00:00:00", "model": "Qwen/Qwen3-32B"},
        "completed": True,
        "partial": False,
        "api_calls": 3,
        "toolsets_used": ["coding"],
        "tool_stats": {},
        "tool_error_counts": {},
    }


def _sharegpt_record_partial() -> dict:
    """A trajectory that hermes flagged partial (invalid tool call mid-loop)."""
    return {
        "prompt_index": 1,
        "conversations": [
            {"from": "system", "value": "system prompt"},
            {"from": "human", "value": "do the thing"},
            {
                "from": "gpt",
                "value": (
                    "<think>\nplan\n</think>\n"
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
        ],
        "completed": False,
        "partial": True,
    }


def _sharegpt_record_max_steps() -> dict:
    """A trajectory that ran out of iterations — final gpt turn still has a tool_call."""
    return {
        "prompt_index": 2,
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "q"},
            {
                "from": "gpt",
                "value": (
                    "<think>\n</think>\n"
                    "<tool_call>\n"
                    '{"name": "calculator", "arguments": {"expression": "1+1"}}\n'
                    "</tool_call>"
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "c1", "name": "calculator", "content": "2"}\n'
                    "</tool_response>"
                ),
            },
            {
                "from": "gpt",
                "value": (
                    "<think>\n</think>\n"
                    "<tool_call>\n"
                    '{"name": "calculator", "arguments": {"expression": "1+1"}}\n'
                    "</tool_call>"
                ),
            },
        ],
        "completed": False,
        "partial": False,
    }


# --- _parse_gpt_value / _parse_tool_value -----------------------------------


def test_parse_gpt_value_extracts_single_tool_call(runner: ModuleType) -> None:
    value = (
        "<think>\nplan\n</think>\n"
        "<tool_call>\n"
        '{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n'
        "</tool_call>"
    )
    calls = runner._parse_gpt_value(value)
    assert calls == [{"name": "read_file", "arguments": {"path": "/tmp/x"}}]


def test_parse_gpt_value_extracts_multiple_tool_calls(runner: ModuleType) -> None:
    value = (
        "<tool_call>\n"
        '{"name": "calculator", "arguments": {"expression": "1+1"}}\n'
        "</tool_call>\n"
        "<tool_call>\n"
        '{"name": "calculator", "arguments": {"expression": "2+2"}}\n'
        "</tool_call>"
    )
    calls = runner._parse_gpt_value(value)
    assert len(calls) == 2
    assert calls[0]["name"] == "calculator"
    assert calls[1]["arguments"]["expression"] == "2+2"


def test_parse_gpt_value_rejects_malformed_tool_call_json(runner: ModuleType) -> None:
    value = "<tool_call>\n{not json}\n</tool_call>"
    with pytest.raises(ValueError, match="malformed <tool_call>"):
        runner._parse_gpt_value(value)


def test_parse_gpt_value_rejects_tool_call_missing_name(runner: ModuleType) -> None:
    value = '<tool_call>\n{"arguments": {}}\n</tool_call>'
    with pytest.raises(ValueError, match="name"):
        runner._parse_gpt_value(value)


def test_parse_gpt_value_returns_empty_for_no_tags(runner: ModuleType) -> None:
    assert runner._parse_gpt_value("<think>\n</think>\nFinal answer text.") == []


def test_parse_tool_value_extracts_responses(runner: ModuleType) -> None:
    value = (
        "<tool_response>\n"
        '{"tool_call_id": "c1", "name": "read_file", "content": "foo"}\n'
        "</tool_response>\n"
        "<tool_response>\n"
        '{"tool_call_id": "c2", "name": "python_eval", "content": "120"}\n'
        "</tool_response>"
    )
    blobs = runner._parse_tool_value(value)
    assert len(blobs) == 2
    assert blobs[0]["content"] == "foo"
    assert blobs[1]["tool_call_id"] == "c2"


def test_parse_tool_value_rejects_malformed_response(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="malformed <tool_response>"):
        runner._parse_tool_value("<tool_response>\n{not json\n</tool_response>")


# --- adapt_hermes_sharegpt_trajectory (full record) --------------------------


def test_adapt_sharegpt_builds_full_trajectory(runner: ModuleType) -> None:
    rec = _sharegpt_record_with_two_tool_calls()
    traj = runner.adapt_hermes_sharegpt_trajectory(
        rec,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:hermes-qwen3-32b-bf16",
    )
    assert isinstance(traj, AgentTrajectory)
    assert traj.task_id == "code-fix-factorial"
    # 3 gpt turns + 2 tool turns = 5 steps; system+human dropped.
    assert len(traj.steps) == 5
    assert [s.role for s in traj.steps] == [
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]


def test_adapt_sharegpt_backfills_tool_results_into_assistant_step(
    runner: ModuleType,
) -> None:
    """The ToolCallRecord on the prior assistant step must carry the
    result_text from the matching <tool_response> block."""
    rec = _sharegpt_record_with_two_tool_calls()
    traj = runner.adapt_hermes_sharegpt_trajectory(
        rec,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assistants = traj.assistant_steps()
    assert assistants[0].tool_calls[0].name == "read_file"
    assert "def factorial" in assistants[0].tool_calls[0].result_text
    assert assistants[1].tool_calls[0].name == "python_eval"
    assert assistants[1].tool_calls[0].result_text == "120"


def test_adapt_sharegpt_terminal_reason_answered(runner: ModuleType) -> None:
    rec = _sharegpt_record_with_two_tool_calls()
    traj = runner.adapt_hermes_sharegpt_trajectory(
        rec,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assert traj.terminal_reason == "answered"
    # <think> blocks must be stripped from the final_answer.
    assert "<think>" not in traj.final_answer
    assert "factorial(5) == 120" in traj.final_answer


def test_adapt_sharegpt_terminal_reason_error_on_partial(
    runner: ModuleType,
) -> None:
    traj = runner.adapt_hermes_sharegpt_trajectory(
        _sharegpt_record_partial(),
        task={"task_id": "t1", "task_text": "do thing"},
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assert traj.terminal_reason == "error"


def test_adapt_sharegpt_terminal_reason_max_steps(runner: ModuleType) -> None:
    traj = runner.adapt_hermes_sharegpt_trajectory(
        _sharegpt_record_max_steps(),
        task={"task_id": "t1", "task_text": "do thing"},
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assert traj.terminal_reason == "max_steps"


def test_adapt_sharegpt_rejects_non_dict_record(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="dict"):
        runner.adapt_hermes_sharegpt_trajectory(
            "not a dict",  # type: ignore[arg-type]
            task=_task(),
            model_id="Qwen/Qwen3-32B",
            engine_label="hermes-qwen3-32b-bf16",
            engine_fingerprint="sha256:x",
        )


def test_adapt_sharegpt_rejects_missing_conversations(runner: ModuleType) -> None:
    with pytest.raises(ValueError, match="conversations"):
        runner.adapt_hermes_sharegpt_trajectory(
            {"prompt_index": 0, "completed": True},
            task=_task(),
            model_id="Qwen/Qwen3-32B",
            engine_label="hermes-qwen3-32b-bf16",
            engine_fingerprint="sha256:x",
        )


def test_adapt_sharegpt_rejects_unsupported_from(runner: ModuleType) -> None:
    rec = {
        "conversations": [
            {"from": "user", "value": "wrong role"},
        ],
        "completed": True,
    }
    with pytest.raises(ValueError, match="unsupported conversations 'from'"):
        runner.adapt_hermes_sharegpt_trajectory(
            rec,
            task=_task(),
            model_id="Qwen/Qwen3-32B",
            engine_label="hermes-qwen3-32b-bf16",
            engine_fingerprint="sha256:x",
        )


def test_adapt_sharegpt_roundtrips_through_json(runner: ModuleType, tmp_path: Path) -> None:
    """The AgentTrajectory produced from a ShareGPT record must serialise
    and re-parse cleanly through Pydantic."""
    rec = _sharegpt_record_with_two_tool_calls()
    traj = runner.adapt_hermes_sharegpt_trajectory(
        rec,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    out = tmp_path / "sharegpt-traj.json"
    out.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
    reloaded = AgentTrajectory.model_validate_json(out.read_text(encoding="utf-8"))
    assert reloaded.task_id == traj.task_id
    assert reloaded.terminal_reason == "answered"
    assert len(reloaded.steps) == 5
    assert reloaded.tool_call_signature() == traj.tool_call_signature()


def test_adapt_record_threads_probe_fields_onto_assistant_step(
    runner: ModuleType,
) -> None:
    """Rollout-side probe fields (prompt/response tokens, logprobs,
    routed experts) on an assistant record must land verbatim on the
    AgentStep. The validator already enforces length parity; we just
    care that the values survive the adapter."""
    rec = {
        "role": "assistant",
        "content": "The answer is 4.",
        "response_token_count": 5,
        "prompt_token_ids": [101, 102, 103, 104],
        "response_token_ids": [201, 202, 203, 204, 205],
        "response_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
        "response_routed_experts": [[[0, 1]]] * 5,
    }
    step = runner.adapt_hermes_record(rec)
    assert step.prompt_token_ids == [101, 102, 103, 104]
    assert step.response_token_ids == [201, 202, 203, 204, 205]
    assert step.response_logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]
    assert step.response_routed_experts == [[[0, 1]]] * 5


def test_adapt_record_skips_absent_probe_fields(runner: ModuleType) -> None:
    """A record with no probe fields must leave them None on the AgentStep
    (back-compat for legacy trajectories captured pre-cycle-3)."""
    step = runner.adapt_hermes_record(
        {"role": "assistant", "content": "hi", "response_token_count": 1}
    )
    assert step.prompt_token_ids is None
    assert step.response_token_ids is None
    assert step.response_logprobs is None
    assert step.response_routed_experts is None


def test_adapt_record_tool_role_rejects_probe_fields(runner: ModuleType) -> None:
    """The AgentStep validator forbids probes on tool-role steps; the
    adapter must surface that as a ValueError instead of swallowing it."""
    with pytest.raises(ValueError, match="tool-role"):
        runner.adapt_hermes_record(
            {
                "role": "tool",
                "tool_call_id": "c1",
                "content": "120",
                # tool-role records should never carry probes; if they do,
                # surface the error rather than silently dropping the data.
                "response_token_ids": [1, 2, 3],
            }
        )


def test_adapt_jsonl_propagates_probe_fields(runner: ModuleType) -> None:
    """End-to-end: probe fields on every assistant record should land
    on every assistant step of the resulting AgentTrajectory."""
    records = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "Let me check.",
            "response_token_count": 3,
            "prompt_token_ids": [10, 11, 12],
            "response_token_ids": [20, 21, 22],
            "response_logprobs": [-0.1, -0.2, -0.3],
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"expression": "2+2"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "4"},
        {
            "role": "assistant",
            "content": "Answer: 4",
            "response_token_count": 3,
            "prompt_token_ids": [10, 11, 12, 13, 14, 15],
            "response_token_ids": [30, 31, 32],
            "response_logprobs": [-0.05, -0.1, -0.15],
        },
    ]
    traj = runner.adapt_hermes_jsonl(
        records,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assistants = traj.assistant_steps()
    assert len(assistants) == 2
    assert assistants[0].response_token_ids == [20, 21, 22]
    assert assistants[0].response_logprobs == [-0.1, -0.2, -0.3]
    # Turn-2 prompt is longer because it includes prior history.
    assert assistants[1].prompt_token_ids == [10, 11, 12, 13, 14, 15]
    assert assistants[1].response_token_ids == [30, 31, 32]
    # Tool-role step must remain probe-free.
    tool_steps = [s for s in traj.steps if s.role == "tool"]
    assert tool_steps[0].response_token_ids is None


def test_adapt_sharegpt_threads_probes_onto_each_assistant_step(
    runner: ModuleType,
) -> None:
    """Hermes-agent's ShareGPT format may surface probes as
    sibling-fields next to ``value`` on each ``from=gpt`` conv entry.
    The adapter must pull them off and put them on the matching
    AgentStep, including after the tool-result backfill rewrites the
    step in place."""
    rec = {
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
                "prompt_token_ids": [10, 11, 12],
                "response_token_ids": [20, 21, 22],
                "response_logprobs": [-0.1, -0.2, -0.3],
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "c1", "name": "calculator", "content": "4"}\n'
                    "</tool_response>"
                ),
            },
            {
                "from": "gpt",
                "value": "<think>\n</think>\nAnswer: 4",
                "prompt_token_ids": [10, 11, 12, 13, 14, 15, 16, 17],
                "response_token_ids": [40, 41, 42, 43],
                "response_logprobs": [-0.05, -0.1, -0.15, -0.2],
            },
        ],
        "completed": True,
        "partial": False,
    }
    traj = runner.adapt_hermes_sharegpt_trajectory(
        rec,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    assistants = traj.assistant_steps()
    assert len(assistants) == 2
    # First turn: probes survived the tool-result backfill rewrite.
    assert assistants[0].response_token_ids == [20, 21, 22]
    assert assistants[0].response_logprobs == [-0.1, -0.2, -0.3]
    assert assistants[0].prompt_token_ids == [10, 11, 12]
    # Second turn carries the accumulated history in prompt_token_ids.
    assert assistants[1].prompt_token_ids == [10, 11, 12, 13, 14, 15, 16, 17]
    assert assistants[1].response_token_ids == [40, 41, 42, 43]
    # JSON round-trip preserves probes.
    reloaded = AgentTrajectory.model_validate_json(traj.model_dump_json())
    assert reloaded.assistant_steps()[0].response_logprobs == [-0.1, -0.2, -0.3]


def test_max_steps_default_is_25() -> None:
    """Per STEER cycle-3 v2: multi-turn agent runs need 15-25 turns
    before answering. The pre-cycle-3 default of 8 truncated long
    trajectories prematurely; the new default is 25."""
    prior = os.environ.pop("MAX_STEPS", None)
    try:
        mod = _load_runner()
        assert mod.MAX_STEPS == 25
    finally:
        if prior is not None:
            os.environ["MAX_STEPS"] = prior


def test_adapt_sharegpt_handles_dict_tool_response_content(
    runner: ModuleType,
) -> None:
    """When a <tool_response> 'content' field is itself a JSON object,
    the adapter must serialise it back to a string for AgentStep.content."""
    rec = {
        "conversations": [
            {"from": "system", "value": "sys"},
            {"from": "human", "value": "q"},
            {
                "from": "gpt",
                "value": (
                    '<tool_call>\n{"name": "web_search", "arguments": {"query": "x"}}\n</tool_call>'
                ),
            },
            {
                "from": "tool",
                "value": (
                    "<tool_response>\n"
                    '{"tool_call_id": "c1", "name": "web_search", "content": {"results": ["a", "b"]}}\n'
                    "</tool_response>"
                ),
            },
            {"from": "gpt", "value": "<think>\n</think>\nFound a and b."},
        ],
        "completed": True,
        "partial": False,
    }
    traj = runner.adapt_hermes_sharegpt_trajectory(
        rec,
        task=_task(),
        model_id="Qwen/Qwen3-32B",
        engine_label="hermes-qwen3-32b-bf16",
        engine_fingerprint="sha256:x",
    )
    tool_steps = [s for s in traj.steps if s.role == "tool"]
    assert len(tool_steps) == 1
    assert "results" in tool_steps[0].content
    # The dict was serialised to JSON, so reparsing it must round-trip.
    assert json.loads(tool_steps[0].content) == {"results": ["a", "b"]}
