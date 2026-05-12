"""Offline tests for scripts/live/hermes_token_extraction.py.

The chat-templating tokenizer is the load-bearing dependency for the
extractor, so the tests stub it with a deterministic ``FakeTokenizer``
that maps every conversation to a sequence of integer token IDs
keyed by role + content. This lets us assert structural invariants
(prompt prefix of full, one pair per assistant turn, history reflects
prior turns + tool responses) without pulling Qwen3 weights or a live
transformers install.

Coverage:
  * single-turn trajectory (no tools)
  * two-turn trajectory (assistant → tool → assistant)
  * four-turn trajectory with multiple tool calls
  * <think> stripping in both response and history
  * rollouts.json shape parity with the consumers in scripts/live/
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint


# --- module loader ----------------------------------------------------------


def _load_extractor() -> ModuleType:
    """Import scripts/live/hermes_token_extraction.py by file path."""
    spec_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "live" / "hermes_token_extraction.py"
    )
    spec = importlib.util.spec_from_file_location("hermes_token_extraction_test", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def extractor() -> ModuleType:
    return _load_extractor()


# --- fake tokenizer ---------------------------------------------------------


class FakeTokenizer:
    """Deterministic chat-templating stub.

    Each role contributes a stable opener token + a UTF-8 codepoint
    sequence for its content + a closer token. ``add_generation_prompt``
    appends the assistant opener so prompt encodings end at the
    generation boundary the same way Qwen3's template does.
    """

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
        # tokenize=False would normally yield a string; the extractor
        # always passes tokenize=True so we ignore the flag and always
        # return list[int]. Every assistant message — both historical
        # ones and a trailing one — gets the GEN_PROMPT_MARKER right
        # after its role-open token, so the encoding of an N-message
        # conversation with add_generation_prompt=False is always a
        # strict superset of the encoding of its prior N-1 messages
        # with add_generation_prompt=True. That mirrors Qwen3's real
        # chat template, where the assistant header is identical in
        # both the prompt and the full-conversation encodings.
        assert tokenize is True, "extractor must call with tokenize=True"
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


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


# --- trajectory fixtures ----------------------------------------------------


def _engine() -> EngineFingerprint:
    return EngineFingerprint(
        name="hermes-qwen3-32b-bf16",
        version="0.13.0",
        fingerprint="sha256:test-fingerprint",
    )


def _make_trajectory(
    task_id: str,
    task_text: str,
    steps: list[AgentStep],
    *,
    terminal_reason: str = "answered",
) -> AgentTrajectory:
    return AgentTrajectory(
        run_id=f"test-{task_id}",
        task_id=task_id,
        task_text=task_text,
        model_id="Qwen/Qwen3-32B",
        rollout_engine=_engine(),
        available_tools=["read_file", "python_eval"],
        steps=steps,
        success=None,
        terminal_reason=terminal_reason,  # type: ignore[arg-type]
        total_assistant_tokens=0,
        final_answer="",
    )


@pytest.fixture
def single_turn() -> AgentTrajectory:
    """Trajectory with one assistant turn that answers from memory."""
    return _make_trajectory(
        "trivia",
        "What is 2 + 2?",
        [
            AgentStep(
                step_idx=0,
                role="assistant",
                content="<think>simple add</think>The answer is 4.",
            )
        ],
        terminal_reason="answered",
    )


@pytest.fixture
def two_turn_with_tool() -> AgentTrajectory:
    """assistant (read_file tool_call) → tool result → assistant answer."""
    return _make_trajectory(
        "code-read",
        "Read /tmp/clip.py and tell me what the function returns for x=42.",
        [
            AgentStep(
                step_idx=0,
                role="assistant",
                content=(
                    "<think>I should inspect the file first.</think>"
                    '<tool_call>{"name": "read_file", '
                    '"arguments": {"path": "/tmp/clip.py"}}</tool_call>'
                ),
                tool_calls=[
                    ToolCallRecord.from_call(
                        name="read_file",
                        arguments={"path": "/tmp/clip.py"},
                        result="def clip(x): return max(0, min(100, x))",
                    )
                ],
            ),
            AgentStep(
                step_idx=0,
                role="tool",
                content="def clip(x): return max(0, min(100, x))",
            ),
            AgentStep(
                step_idx=1,
                role="assistant",
                content=(
                    "<think>clip clamps to [0,100], so 42 stays 42.</think>The function returns 42."
                ),
            ),
        ],
        terminal_reason="answered",
    )


@pytest.fixture
def four_turn_multi_tool() -> AgentTrajectory:
    """assistant(tool) → tool → assistant(tool) → tool → assistant(tool) → tool → assistant."""
    return _make_trajectory(
        "code-verify",
        (
            "Read /tmp/fib.py, identify the off-by-one bug, "
            "verify the fix with python_eval, then answer fib(10)."
        ),
        [
            AgentStep(
                step_idx=0,
                role="assistant",
                content=(
                    "<think>Start by reading the file.</think>"
                    '<tool_call>{"name": "read_file", '
                    '"arguments": {"path": "/tmp/fib.py"}}</tool_call>'
                ),
                tool_calls=[
                    ToolCallRecord.from_call(
                        name="read_file",
                        arguments={"path": "/tmp/fib.py"},
                        result="def fib(n):\n  if n <= 1: return 1\n  ...",
                    )
                ],
            ),
            AgentStep(
                step_idx=0,
                role="tool",
                content="def fib(n):\n  if n <= 1: return 1\n  return fib(n-1) + fib(n-2)",
            ),
            AgentStep(
                step_idx=1,
                role="assistant",
                content=(
                    "<think>Base case is off; should be n == 0.</think>"
                    '<tool_call>{"name": "python_eval", '
                    '"arguments": {"code": "def fib(n): '
                    'return 0 if n==0 else 1 if n==1 else fib(n-1)+fib(n-2)\\nprint(fib(10))"}}'
                    "</tool_call>"
                ),
                tool_calls=[
                    ToolCallRecord.from_call(
                        name="python_eval",
                        arguments={"code": "print(fib(10))"},
                        result="55",
                    )
                ],
            ),
            AgentStep(step_idx=1, role="tool", content="55"),
            AgentStep(
                step_idx=2,
                role="assistant",
                content=(
                    "<think>Double-check.</think>"
                    '<tool_call>{"name": "python_eval", '
                    '"arguments": {"code": "print(fib(10))"}}</tool_call>'
                ),
                tool_calls=[
                    ToolCallRecord.from_call(
                        name="python_eval",
                        arguments={"code": "print(fib(10))"},
                        result="55",
                    )
                ],
            ),
            AgentStep(step_idx=2, role="tool", content="55"),
            AgentStep(
                step_idx=3,
                role="assistant",
                content=("<think>Confirmed.</think>fib(10) is 55."),
            ),
        ],
        terminal_reason="answered",
    )


# --- tests ------------------------------------------------------------------


def test_single_turn_yields_one_pair(extractor, tokenizer, single_turn):
    pairs = extractor.extract_token_pairs(single_turn, tokenizer)
    assert len(pairs) == 1
    prompt_ids, response_ids = pairs[0]
    # Prompt ends at the assistant generation-prompt marker.
    assert prompt_ids[-1] == FakeTokenizer.GEN_PROMPT_MARKER
    # Prompt contains the user task tokens.
    assert FakeTokenizer.ROLE_OPEN["user"] in prompt_ids
    # Response is non-empty and does not include the <think> block.
    assert len(response_ids) > 0
    # <think> content "simple add" must not appear as a substring of the
    # response: the response text after _strip_think is "The answer is 4."
    response_text = "".join(chr(t) for t in response_ids if t < 128)
    assert "simple add" not in response_text
    assert "The answer is 4." in response_text


def test_two_turn_with_tool_builds_history(extractor, tokenizer, two_turn_with_tool):
    pairs = extractor.extract_token_pairs(two_turn_with_tool, tokenizer)
    assert len(pairs) == 2

    p0, r0 = pairs[0]
    p1, r1 = pairs[1]

    # Turn 1's prompt must be strictly longer than turn 0's: it carries
    # the assistant turn 0 content + the tool response in between.
    assert len(p1) > len(p0)

    # Turn 1's prompt must include exactly one assistant block and one
    # tool block from history. Counts of role-open tokens give us that.
    def count(seq: list[int], tok: int) -> int:
        return sum(1 for t in seq if t == tok)

    assert count(p0, FakeTokenizer.ROLE_OPEN["assistant"]) == 1
    # ^ assistant role-open appears once via the generation prompt only.
    assert count(p1, FakeTokenizer.ROLE_OPEN["assistant"]) == 2
    # ^ prior assistant turn + generation prompt header.
    assert count(p1, FakeTokenizer.ROLE_OPEN["tool"]) == 1

    # Response 1 contains the second assistant's text ("returns 42").
    response_text = "".join(chr(t) for t in r1 if t < 128)
    assert "returns 42" in response_text
    assert "<think>" not in response_text


def test_four_turn_yields_four_pairs(extractor, tokenizer, four_turn_multi_tool):
    pairs = extractor.extract_token_pairs(four_turn_multi_tool, tokenizer)
    assert len(pairs) == 4

    def count(seq: list[int], tok: int) -> int:
        return sum(1 for t in seq if t == tok)

    # Each successive prompt must include one more prior assistant + tool
    # pair than the previous one (history grows monotonically).
    assistant_open = FakeTokenizer.ROLE_OPEN["assistant"]
    tool_open = FakeTokenizer.ROLE_OPEN["tool"]
    expected_assistant_counts = [1, 2, 3, 4]  # gen-prompt + i prior turns
    expected_tool_counts = [0, 1, 2, 3]
    for i, (p, _r) in enumerate(pairs):
        assert count(p, assistant_open) == expected_assistant_counts[i], (
            f"turn {i}: assistant-open count mismatch"
        )
        assert count(p, tool_open) == expected_tool_counts[i], f"turn {i}: tool-open count mismatch"

    # Final answer ("fib(10) is 55.") must show up only in the last
    # response, never in the preceding history.
    last_response = "".join(chr(t) for t in pairs[-1][1] if t < 128)
    assert "fib(10) is 55." in last_response
    for i, (p, _r) in enumerate(pairs[:-1]):
        prompt_text = "".join(chr(t) for t in p if t < 128)
        assert "fib(10) is 55." not in prompt_text, (
            f"turn {i} prompt should not yet include the final answer"
        )


def test_history_strips_think_blocks(extractor, tokenizer, two_turn_with_tool):
    pairs = extractor.extract_token_pairs(two_turn_with_tool, tokenizer)
    _p0, _r0 = pairs[0]
    p1, _r1 = pairs[1]
    history_text = "".join(chr(t) for t in p1 if t < 128)
    # Prior assistant's <think> body must NOT leak into the next prompt.
    assert "I should inspect" not in history_text
    # But the <tool_call> XML body MUST be preserved (it's the action
    # the model took, not chain-of-thought).
    assert "read_file" in history_text


def test_rollouts_shape_matches_trainer_consumers(extractor, tokenizer, four_turn_multi_tool):
    rollouts = extractor.trajectory_to_rollouts(four_turn_multi_tool, tokenizer)
    assert len(rollouts) == 4
    for i, entry in enumerate(rollouts):
        # Keys that run_fsdp_reference.py / run_fsdp_moe_reference.py /
        # run_megatron_moe_reference.py read directly:
        assert entry["model"] == "Qwen/Qwen3-32B"
        assert entry["prompt_idx"] == i
        assert isinstance(entry["prompt_token_ids"], list)
        assert isinstance(entry["response_token_ids"], list)
        assert all(isinstance(t, int) for t in entry["prompt_token_ids"])
        assert all(isinstance(t, int) for t in entry["response_token_ids"])
        # Provenance keys for back-referencing reports to trajectories:
        assert entry["task_id"] == "code-verify"
        assert entry["assistant_turn_idx"] == i
        assert entry["trajectory_run_id"].startswith("test-")


def test_extract_with_system_prompt_prepends_system_role(extractor, tokenizer, single_turn):
    pairs = extractor.extract_token_pairs(
        single_turn, tokenizer, system_prompt="You are a helpful assistant."
    )
    p0, _r0 = pairs[0]
    assert FakeTokenizer.ROLE_OPEN["system"] in p0
    # Sanity: the prompt without a system prompt does not contain it.
    pairs_no_sys = extractor.extract_token_pairs(single_turn, tokenizer)
    p0_no_sys, _ = pairs_no_sys[0]
    assert FakeTokenizer.ROLE_OPEN["system"] not in p0_no_sys
    assert len(p0) > len(p0_no_sys)


def test_extract_prefers_captured_token_ids_over_chat_template(extractor):
    """When AgentStep carries rollout-probe IDs, extract returns them
    verbatim — no tokenizer is needed.

    This is the long-term stable path: vLLM emits the exact prompt /
    response token IDs at generation time, the trajectory persists
    them, and the trainer-side teacher-force replays the exact same
    IDs without any chat-template re-encoding (which can drift across
    tokenizer / transformers versions).
    """
    captured_prompt = [101, 102, 103, 104, 105]
    captured_response = [201, 202, 203]
    traj = _make_trajectory(
        "captured",
        "what is 2+2?",
        [
            AgentStep(
                step_idx=0,
                role="assistant",
                content="The answer is 4.",
                prompt_token_ids=captured_prompt,
                response_token_ids=captured_response,
                response_logprobs=[-0.1, -0.2, -0.3],
            )
        ],
    )
    # tokenizer=None proves the path doesn't need one when IDs are captured.
    pairs = extractor.extract_token_pairs(traj, tokenizer=None)
    assert pairs == [(captured_prompt, captured_response)]


def test_extract_raises_when_no_tokens_and_no_tokenizer(extractor):
    traj = _make_trajectory(
        "no-capture",
        "hi",
        [AgentStep(step_idx=0, role="assistant", content="hello")],
    )
    with pytest.raises(ValueError, match="no captured token IDs"):
        extractor.extract_token_pairs(traj, tokenizer=None)


def test_extract_mixed_capture_falls_back_to_chat_template(extractor, tokenizer):
    """If only some assistant turns have captured IDs, the rest
    fall back to chat-template re-encoding using the supplied tokenizer.

    A real-world case: a trajectory might be re-played from a
    legacy capture for turn 0 (text only) but turn 1 onward came
    from a probe-enabled re-run. The function must not refuse the
    mixed input — it processes each turn under the best available
    signal.
    """
    captured_prompt = [101, 102, 103]
    captured_response = [201, 202]
    traj = _make_trajectory(
        "mixed",
        "two turns",
        [
            AgentStep(step_idx=0, role="assistant", content="first turn"),
            AgentStep(step_idx=0, role="tool", content="tool result"),
            AgentStep(
                step_idx=1,
                role="assistant",
                content="second turn answer",
                prompt_token_ids=captured_prompt,
                response_token_ids=captured_response,
            ),
        ],
    )
    pairs = extractor.extract_token_pairs(traj, tokenizer)
    assert len(pairs) == 2
    # Turn 0 came through the chat-template fallback.
    assert pairs[0][0][-1] == FakeTokenizer.GEN_PROMPT_MARKER
    # Turn 1 returned the captured IDs verbatim.
    assert pairs[1] == (captured_prompt, captured_response)


def test_coerce_handles_batch_encoding_and_encoding_shapes(extractor):
    """Verify the BatchEncoding / Encoding shape adapters used on the spot.

    HuggingFace's PreTrainedTokenizerFast returns a ``BatchEncoding``
    (dict-like with ``.input_ids``) or a length-1 list of
    ``tokenizers.Encoding`` (which carries ``.ids``) when
    ``apply_chat_template(..., tokenize=True)`` is called. The extractor
    has to normalise those into ``list[int]`` so its slicing arithmetic
    is shape-agnostic.
    """

    class _BatchEncodingLike:
        def __init__(self, ids: list[int]):
            self.input_ids = ids

    class _BatchEncodingLikeNested:
        def __init__(self, ids: list[int]):
            self.input_ids = [ids]

    class _EncodingLike:
        def __init__(self, ids: list[int]):
            self.ids = ids

    coerce = extractor._coerce_to_id_list

    assert coerce([1, 2, 3]) == [1, 2, 3]
    assert coerce((4, 5)) == [4, 5]
    assert coerce(_BatchEncodingLike([10, 20, 30])) == [10, 20, 30]
    assert coerce(_BatchEncodingLikeNested([10, 20, 30])) == [10, 20, 30]
    assert coerce(_EncodingLike([7, 8, 9])) == [7, 8, 9]
    assert coerce([_EncodingLike([7, 8, 9])]) == [7, 8, 9]
    assert coerce([[1, 2, 3]]) == [1, 2, 3]

    with pytest.raises(ValueError, match="unsupported shape"):
        coerce(object())


def test_empty_trajectory_yields_no_pairs(extractor, tokenizer):
    traj = _make_trajectory(
        "empty",
        "What's the meaning of life?",
        [AgentStep(step_idx=0, role="tool", content="42")],
        terminal_reason="error",
    )
    # No assistant turns at all → no pairs.
    pairs = extractor.extract_token_pairs(traj, tokenizer)
    assert pairs == []
    rollouts = extractor.trajectory_to_rollouts(traj, tokenizer)
    assert rollouts == []
