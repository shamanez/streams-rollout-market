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
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
ENGINE_VERSION = os.environ.get("ENGINE_VERSION", "0.20.1")


def _default_engine_label() -> str:
    """Derive `hermes-{model_slug}-{precision}` from MODEL + PRECISION."""
    slug = MODEL.split("/")[-1].lower().replace("_", "-")
    return f"hermes-{slug}-{PRECISION}"


ENGINE_LABEL = os.environ.get("ENGINE_LABEL", _default_engine_label())
ENGINE_FINGERPRINT = os.environ.get(
    "ENGINE_FINGERPRINT", f"sha256:{ENGINE_LABEL}-tp4-l40s"
)
OUT_DIR = Path(
    os.environ.get(
        "OUT_DIR",
        f"runs/live/agent/hermes/{MODEL.split('/')[-1].lower()}/{PRECISION}",
    )
)


# --- adapter (pure) ---------------------------------------------------------


_VALID_ROLES = ("assistant", "tool")


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
    """
    if not isinstance(record, dict):
        raise ValueError(
            f"hermes record must be a dict, got {type(record).__name__}: {record!r}"
        )
    role = record.get("role")
    if role not in _VALID_ROLES:
        raise ValueError(
            f"hermes record role must be one of {_VALID_ROLES}, got {role!r}"
        )

    content = record.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        raise ValueError(
            f"hermes record 'content' must be a string, got {type(content).__name__}"
        )

    if role == "tool":
        return AgentStep(step_idx=step_idx, role="tool", content=content)

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
            raise ValueError(
                f"hermes tool_call entry must be a dict, got {type(call).__name__}"
            )
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
        captured.append(
            ToolCallRecord.from_call(name=name, arguments=args, result=result)
        )

    return AgentStep(
        step_idx=step_idx,
        role="assistant",
        content=content,
        tool_calls=captured,
        finish_reason=record.get("finish_reason"),
        response_token_count=int(record.get("response_token_count") or 0),
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
        raise ValueError(
            f"records must be a list, got {type(records).__name__}"
        )

    # Pass 1: build tool_call_id → result text.
    tool_results: dict[str, str] = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(
                f"hermes record must be a dict, got {type(rec).__name__}: {rec!r}"
            )
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
            raise ValueError(
                f"hermes JSONL line {i + 1} is not valid JSON: {exc}"
            ) from exc
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
                print(
                    f"[error] {task_id}: hermes CLI exited rc={rc}", file=sys.stderr
                )
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
