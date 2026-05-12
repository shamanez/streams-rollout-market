"""Mac-side wrapper around the NousResearch/hermes-agent CLI.

Drives the *real* Hermes Agent harness against an OpenAI-compatible
endpoint (typically our spot vLLM serve over an SSH tunnel) for each
task in ``scripts/live/agent_tasks_hermes.json``, parses its JSONL
trajectory output, and writes a Pydantic ``AgentTrajectory`` JSON per
task under ``OUT_DIR``.

The piece that earns its keep here is ``adapt_hermes_record``: a pure
function that converts ONE Hermes JSONL record into one ``AgentStep``,
testable offline against a fixture. ``adapt_hermes_jsonl`` is the
orchestration wrapper that pairs tool-call IDs with their tool-result
records and emits the full trajectory.

Usage::

    OPENAI_BASE_URL=http://localhost:8000/v1 \\
    OPENAI_API_KEY=dummy \\
    MODEL=Qwen/Qwen3-32B \\
    PRECISION=bf16 \\
    TASKS_FILE=scripts/live/agent_tasks_hermes.json \\
    OUT_DIR=runs/live/agent/hermes/qwen3-32b/bf16 \\
    python scripts/live/hermes_agent_runner.py

The Hermes Agent CLI invocation is parametrised via env so the runner
also works against any locally installed Hermes-compatible binary
(e.g. ``hermes-agent`` on PATH). The OpenAI-format trajectory JSONL
schema this adapter consumes is::

    {"role": "assistant", "content": "<think>...</think>"}
    {"role": "assistant", "content": "I'll search.",
     "tool_calls": [{"id": "c1", "type": "function",
                     "function": {"name": "web_search",
                                  "arguments": "{\\"query\\":\\"x\\"}"}}]}
    {"role": "tool", "tool_call_id": "c1", "content": "result text"}
    {"role": "assistant", "content": "The answer is 4."}

System and user records (the prompt envelope) are tolerated and
skipped; only assistant/tool records become trajectory steps.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint


# --- env config -------------------------------------------------------------

OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "dummy")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
MODEL_ID = os.environ.get("MODEL_ID", MODEL)
PRECISION = os.environ.get("PRECISION", "bf16")
TASKS_FILE = os.environ.get("TASKS_FILE", "scripts/live/agent_tasks_hermes.json")
HERMES_CLI = os.environ.get("HERMES_CLI", "hermes-agent")
HERMES_CONFIG = os.environ.get("HERMES_CONFIG", "")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "25"))
ENGINE_VERSION = os.environ.get("ENGINE_VERSION", "0.20.1")


def _default_engine_label() -> str:
    """Derive `hermes-{model_slug}-{precision}` from MODEL + PRECISION."""
    slug = MODEL.split("/")[-1].lower().replace("_", "-")
    return f"hermes-{slug}-{PRECISION}"


ENGINE_LABEL = os.environ.get("ENGINE_LABEL", _default_engine_label())
ENGINE_FINGERPRINT = os.environ.get("ENGINE_FINGERPRINT", f"sha256:{ENGINE_LABEL}-tp4-l40s")
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        f"runs/live/agent/hermes/{MODEL.split('/')[-1].lower()}/{PRECISION}",
    )
)


# --- adapter (pure) ---------------------------------------------------------


_VALID_ROLES = ("assistant", "tool")

# Optional rollout-side probe fields that the Hermes Agent harness may
# stamp onto each assistant record when vLLM was asked to return them
# at generation time. They flow straight through to the AgentStep so
# the trainer-side teacher-force can re-evaluate the exact same token
# pair. Tool-role records must never carry them — the AgentStep
# validator enforces this and we mirror the rule here.
_PROBE_FIELDS = (
    "prompt_token_ids",
    "response_token_ids",
    "response_logprobs",
    "response_routed_experts",
)


def _extract_probes(record: dict) -> dict:
    """Pull probe fields out of a record dict, skipping ones set to None."""
    return {k: record[k] for k in _PROBE_FIELDS if record.get(k) is not None}


def adapt_hermes_record(
    record: dict,
    *,
    tool_results: dict[str, str] | None = None,
    step_idx: int = 0,
) -> AgentStep:
    """Convert ONE hermes-agent JSONL record into an AgentStep.

    Pure function — no I/O, no env reads. ``tool_results`` is a mapping
    from ``tool_call_id`` to the corresponding tool result text; the
    caller is responsible for building it by scanning tool-role records
    first. Records of unsupported role or missing required fields raise
    ``ValueError`` with a message that names the offending field.

    Assistant records may optionally carry the rollout-time probe
    fields (``prompt_token_ids``, ``response_token_ids``,
    ``response_logprobs``, ``response_routed_experts``); they are
    threaded onto the resulting ``AgentStep`` verbatim.
    """
    if not isinstance(record, dict):
        raise ValueError(f"hermes record must be a dict, got {type(record).__name__}: {record!r}")
    role = record.get("role")
    if role not in _VALID_ROLES:
        raise ValueError(f"hermes record role must be one of {_VALID_ROLES}, got {role!r}")

    content = record.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        raise ValueError(f"hermes record 'content' must be a string, got {type(content).__name__}")

    if role == "tool":
        return AgentStep(
            step_idx=step_idx,
            role="tool",
            content=content,
            **_extract_probes(record),
        )

    # assistant role — content + optional tool_calls.
    tool_calls_raw = record.get("tool_calls") or []
    if not isinstance(tool_calls_raw, list):
        raise ValueError(
            f"hermes record 'tool_calls' must be a list, got {type(tool_calls_raw).__name__}"
        )

    results = tool_results or {}
    captured: list[ToolCallRecord] = []
    for call in tool_calls_raw:
        if not isinstance(call, dict):
            raise ValueError(f"hermes tool_call entry must be a dict, got {type(call).__name__}")
        fn = call.get("function") or {}
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"hermes tool_call missing function.name: {call!r}")
        args_raw = fn.get("arguments", "{}")
        if isinstance(args_raw, dict):
            args: dict | str = args_raw
        elif isinstance(args_raw, str):
            try:
                args = json.loads(args_raw)
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
        else:
            raise ValueError(
                f"hermes tool_call function.arguments must be str or dict, got "
                f"{type(args_raw).__name__}"
            )
        call_id = call.get("id") or ""
        result = results.get(call_id, "")
        captured.append(ToolCallRecord.from_call(name=name, arguments=args, result=result))

    return AgentStep(
        step_idx=step_idx,
        role="assistant",
        content=content,
        tool_calls=captured,
        finish_reason=record.get("finish_reason"),
        response_token_count=int(record.get("response_token_count") or 0),
        **_extract_probes(record),
    )


def adapt_hermes_jsonl(
    records: list[dict],
    *,
    task: dict,
    model_id: str,
    engine_label: str,
    engine_fingerprint: str,
    engine_version: str = ENGINE_VERSION,
    available_tools: list[str] | None = None,
    max_steps: int = MAX_STEPS,
) -> AgentTrajectory:
    """Adapt a full Hermes JSONL record list into an AgentTrajectory.

    Two-pass: first pass collects tool_call_id → result text, second
    pass adapts each actionable record (assistant or tool role) via
    ``adapt_hermes_record``. System / user records are tolerated and
    skipped — they belong to the prompt envelope, not the trajectory.

    The ``terminal_reason`` is derived from the last assistant record:
    - no trailing tool_calls → ``"answered"`` (its content is the final answer)
    - trailing tool_calls AND assistant-step count >= ``max_steps`` → ``"max_steps"``
    - trailing tool_calls but assistant-step count < ``max_steps`` → ``"error"``
      (the run exited mid-loop without a terminal answer)
    - empty trajectory → ``"error"``
    """
    if not isinstance(records, list):
        raise ValueError(f"records must be a list, got {type(records).__name__}")

    # Pass 1: build tool_call_id → result text.
    tool_results: dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"hermes record must be a dict, got {type(rec).__name__}: {rec!r}")
        if rec.get("role") == "tool":
            tcid = rec.get("tool_call_id") or ""
            if tcid:
                tool_results[tcid] = str(rec.get("content") or "")

    # Pass 2: adapt assistant/tool records into AgentSteps.
    steps: list[AgentStep] = []
    total_assistant_tokens = 0
    assistant_idx = 0
    for rec in records:
        role = rec.get("role")
        if role not in _VALID_ROLES:
            continue
        idx = assistant_idx if role == "assistant" else max(0, assistant_idx - 1)
        step = adapt_hermes_record(rec, tool_results=tool_results, step_idx=idx)
        steps.append(step)
        if step.role == "assistant":
            total_assistant_tokens += step.response_token_count
            assistant_idx += 1

    assistant_steps = [s for s in steps if s.role == "assistant"]

    if not assistant_steps:
        terminal: str = "error"
        final_answer = ""
    else:
        last = assistant_steps[-1]
        if last.tool_calls:
            if len(assistant_steps) >= max_steps:
                terminal = "max_steps"
            else:
                terminal = "error"
            final_answer = ""
        else:
            terminal = "answered"
            final_answer = last.content

    return AgentTrajectory(
        run_id=f"{engine_label}-{task['task_id']}",
        task_id=task["task_id"],
        task_text=task["task_text"],
        model_id=model_id,
        rollout_engine=EngineFingerprint(
            name=engine_label,
            version=engine_version,
            fingerprint=engine_fingerprint,
        ),
        available_tools=list(available_tools or []),
        steps=steps,
        success=None,
        terminal_reason=terminal,  # type: ignore[arg-type]
        total_assistant_tokens=total_assistant_tokens,
        final_answer=final_answer,
        created_at=datetime.now(timezone.utc),
    )


# --- ShareGPT adapter (real hermes-agent batch_runner.py output) ------------

# Hermes Agent's batch_runner.py emits one JSONL record per prompt, with the
# message history nested under ``conversations`` in ShareGPT shape
# (``from/value``). Assistant turns embed ``<think>`` + zero or more
# ``<tool_call>`` JSON blocks inside ``value``; tool turns embed one or more
# ``<tool_response>`` JSON blocks. The schema is documented in
# ``run_agent.py::_convert_to_trajectory_format`` (Hermes Agent commit
# at clone time).

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_gpt_value(value: str) -> list[dict]:
    """Extract the list of ``<tool_call>`` JSON blobs from a ``from=gpt`` value.

    Returns an ordered list of ``{"name": str, "arguments": dict|str}`` dicts,
    one per ``<tool_call>...</tool_call>`` block. Raises ``ValueError`` if any
    block contains malformed JSON or is missing the ``name`` field.
    """
    out: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(value):
        body = m.group(1).strip()
        try:
            blob = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed <tool_call> JSON: {body[:80]!r} ({exc})") from exc
        if not isinstance(blob, dict) or "name" not in blob:
            raise ValueError(f"<tool_call> body must be a dict with 'name': {blob!r}")
        out.append(blob)
    return out


def _parse_tool_value(value: str) -> list[dict]:
    """Extract the list of ``<tool_response>`` JSON blobs from a ``from=tool`` value.

    Each ``<tool_response>`` body is a JSON object of shape
    ``{"tool_call_id": str, "name": str, "content": ...}``. Returns the blobs
    in source order; raises ``ValueError`` on malformed JSON.
    """
    out: list[dict] = []
    for m in _TOOL_RESPONSE_RE.finditer(value):
        body = m.group(1).strip()
        try:
            blob = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed <tool_response> JSON: {body[:80]!r} ({exc})") from exc
        out.append(blob)
    return out


def adapt_hermes_sharegpt_trajectory(
    record: dict,
    *,
    task: dict,
    model_id: str,
    engine_label: str,
    engine_fingerprint: str,
    engine_version: str = ENGINE_VERSION,
    available_tools: list[str] | None = None,
) -> AgentTrajectory:
    """Adapt ONE hermes-agent ShareGPT trajectory record into an AgentTrajectory.

    The input shape comes directly from
    ``hermes-agent/batch_runner.py``::process_prompt — one top-level JSON object
    per prompt with::

        {"prompt_index": int,
         "conversations": [{"from": "system|human|gpt|tool", "value": str}, ...],
         "completed": bool, "partial": bool,
         "api_calls": int, "tool_stats": {...}, "metadata": {...}}

    ``from=system|human`` records belong to the prompt envelope and are
    dropped. Each ``from=gpt`` record becomes an ``AgentStep(role=assistant)``,
    with ``<tool_call>`` XML blocks parsed into ``ToolCallRecord`` entries
    (result_text initially empty). Each ``from=tool`` record contains one
    ``<tool_response>`` block per outstanding tool call; we emit one
    ``AgentStep(role=tool)`` per response and backfill the matching
    ``ToolCallRecord.result_text`` on the prior assistant step (positional
    pairing — Hermes preserves order; ``tool_call_id`` is also threaded but
    not load-bearing).

    ``terminal_reason`` derives from the record's ``completed`` / ``partial``
    flags plus the trailing assistant turn:

      - ``partial=True``              → ``"error"`` (Hermes stopped on bad output)
      - ``completed=True`` and trailing gpt has no tool_calls → ``"answered"``
      - otherwise                     → ``"max_steps"``
    """
    if not isinstance(record, dict):
        raise ValueError(f"hermes record must be a dict, got {type(record).__name__}")
    convs = record.get("conversations")
    if not isinstance(convs, list):
        raise ValueError(f"hermes record missing 'conversations' list (got {type(convs).__name__})")

    steps: list[AgentStep] = []
    assistant_idx = 0
    pending_calls: list[ToolCallRecord] = []
    pending_assistant_step_pos: int | None = None
    pending_call_names: list[str] = []
    pending_call_args: list[str] = []

    for conv in convs:
        if not isinstance(conv, dict):
            raise ValueError(f"conversations entry must be a dict, got {type(conv).__name__}")
        role_from = conv.get("from")
        value = conv.get("value")
        if not isinstance(value, str):
            raise ValueError(f"conversations entry 'value' must be a string for from={role_from!r}")

        if role_from in ("system", "human"):
            continue

        if role_from == "gpt":
            raw_calls = _parse_gpt_value(value)
            captured: list[ToolCallRecord] = []
            names: list[str] = []
            args_strs: list[str] = []
            for c in raw_calls:
                args = c.get("arguments", {})
                if not isinstance(args, (dict, str)):
                    args = {"_raw": str(args)}
                tc = ToolCallRecord.from_call(name=c["name"], arguments=args, result="")
                captured.append(tc)
                names.append(c["name"])
                args_strs.append(tc.arguments_json)
            steps.append(
                AgentStep(
                    step_idx=assistant_idx,
                    role="assistant",
                    content=value,
                    tool_calls=captured,
                    **_extract_probes(conv),
                )
            )
            pending_calls = captured
            pending_assistant_step_pos = len(steps) - 1
            pending_call_names = names
            pending_call_args = args_strs
            assistant_idx += 1
            continue

        if role_from == "tool":
            blobs = _parse_tool_value(value)
            for k, blob in enumerate(blobs):
                content = blob.get("content", "")
                if isinstance(content, (dict, list)):
                    content_str = json.dumps(content, ensure_ascii=False, sort_keys=True)
                else:
                    content_str = str(content)
                steps.append(
                    AgentStep(
                        step_idx=max(0, assistant_idx - 1),
                        role="tool",
                        content=content_str,
                    )
                )
                if k < len(pending_calls):
                    pending_calls[k] = ToolCallRecord.from_call(
                        name=pending_call_names[k],
                        arguments=(
                            json.loads(pending_call_args[k])
                            if pending_call_args[k].strip().startswith(("{", "["))
                            else pending_call_args[k]
                        ),
                        result=content_str,
                    )
            if pending_assistant_step_pos is not None and pending_calls:
                prev = steps[pending_assistant_step_pos]
                steps[pending_assistant_step_pos] = AgentStep(
                    step_idx=prev.step_idx,
                    role=prev.role,
                    content=prev.content,
                    tool_calls=pending_calls,
                    finish_reason=prev.finish_reason,
                    response_token_count=prev.response_token_count,
                    prompt_token_ids=prev.prompt_token_ids,
                    response_token_ids=prev.response_token_ids,
                    response_logprobs=prev.response_logprobs,
                    response_routed_experts=prev.response_routed_experts,
                )
            pending_calls = []
            pending_assistant_step_pos = None
            pending_call_names = []
            pending_call_args = []
            continue

        raise ValueError(f"unsupported conversations 'from' value: {role_from!r}")

    completed = bool(record.get("completed", False))
    partial = bool(record.get("partial", False))
    assistant_steps = [s for s in steps if s.role == "assistant"]

    if not assistant_steps:
        terminal: str = "error"
        final_answer = ""
    elif partial:
        terminal = "error"
        final_answer = ""
    elif completed and not assistant_steps[-1].tool_calls:
        terminal = "answered"
        final_answer = _THINK_RE.sub("", assistant_steps[-1].content).strip()
    else:
        terminal = "max_steps"
        final_answer = ""

    return AgentTrajectory(
        run_id=f"{engine_label}-{task['task_id']}",
        task_id=task["task_id"],
        task_text=task["task_text"],
        model_id=model_id,
        rollout_engine=EngineFingerprint(
            name=engine_label,
            version=engine_version,
            fingerprint=engine_fingerprint,
        ),
        available_tools=list(available_tools or []),
        steps=steps,
        success=None,
        terminal_reason=terminal,  # type: ignore[arg-type]
        total_assistant_tokens=0,
        final_answer=final_answer,
        created_at=datetime.now(timezone.utc),
    )


# --- CLI orchestration ------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"hermes JSONL line {i + 1} is not valid JSON: {exc}") from exc
        out.append(rec)
    return out


def _run_hermes_cli(task: dict, out_jsonl: Path) -> int:
    """Shell out to the Hermes CLI for one task; return its exit code.

    The CLI is invoked via ``subprocess.run`` so it inherits the
    ``OPENAI_BASE_URL`` + ``OPENAI_API_KEY`` env vars. The exact CLI
    flag surface varies between Hermes versions; we keep it env-driven
    so operators can override ``HERMES_CLI`` (binary name) and
    ``HERMES_CONFIG`` (path to a YAML config) without code changes.
    """
    cmd = [HERMES_CLI, "run", "--task", task["task_text"], "--jsonl-out", str(out_jsonl)]
    if HERMES_CONFIG:
        cmd.extend(["--config", HERMES_CONFIG])
    env = os.environ.copy()
    env.setdefault("OPENAI_BASE_URL", OPENAI_BASE_URL)
    env.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
    env.setdefault("OPENAI_MODEL", MODEL)
    print(f"[hermes-cli] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    return proc.returncode


def main() -> int:
    tasks_path = Path(TASKS_FILE)
    if not tasks_path.exists():
        print(f"[fatal] tasks file not found: {tasks_path}", file=sys.stderr)
        return 2
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"[runner] model={MODEL} precision={PRECISION} "
        f"engine_label={ENGINE_LABEL} out_dir={OUT_DIR}",
        flush=True,
    )

    for task in tasks:
        task_id = task["task_id"]
        out_file = OUT_DIR / f"{task_id}.json"
        if out_file.exists():
            print(f"[skip] {task_id}: already exists at {out_file}", flush=True)
            continue

        t0 = time.time()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, prefix=f"hermes-{task_id}-"
        ) as tf:
            jsonl_path = Path(tf.name)
        try:
            rc = _run_hermes_cli(task, jsonl_path)
            if rc != 0:
                print(f"[error] {task_id}: hermes CLI exited rc={rc}", file=sys.stderr)
                continue
            records = _read_jsonl(jsonl_path)
            traj = adapt_hermes_jsonl(
                records,
                task=task,
                model_id=MODEL_ID,
                engine_label=ENGINE_LABEL,
                engine_fingerprint=ENGINE_FINGERPRINT,
                engine_version=ENGINE_VERSION,
                available_tools=task.get("available_tools", []),
                max_steps=MAX_STEPS,
            )
            out_file.write_text(traj.model_dump_json(indent=2), encoding="utf-8")
            dt = time.time() - t0
            print(
                f"[done] {task_id}: {len(traj.assistant_steps())} assistant steps, "
                f"terminal={traj.terminal_reason}, {dt:.1f}s -> {out_file}",
                flush=True,
            )
        finally:
            try:
                jsonl_path.unlink()
            except OSError:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
