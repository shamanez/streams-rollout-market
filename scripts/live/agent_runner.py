"""Mac-side agent orchestrator against an OpenAI-compatible vLLM/sglang server.

Drives a multi-step tool-using agent loop, captures every step
(assistant decisions + tool invocations) in the AgentTrajectory schema,
and writes one trajectory JSON per (task, engine) pair.

The tools are deterministic simulations: given the same arguments they
return the same output, so two engines running the same task can be
compared step by step. The point is to measure how the AGENT's tool
choices and arguments diverge across engines, not how realistic the
tool outputs are.

Usage::

    BASE_URL=http://localhost:8000/v1 \\
    MODEL=Qwen/Qwen3-32B \\
    ENGINE_LABEL=vllm-bf16 \\
    OUT_DIR=runs/live/agent/vllm-bf16 \\
    python scripts/live/agent_runner.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import operator
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from rollout_market.observatory.agent_trajectory_lab import (
    AgentStep,
    AgentTrajectory,
    ToolCallRecord,
)
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-32B")
# MODEL_ID is the *logical* checkpoint identity carried in the trajectory
# JSON; MODEL is what the served API expects. They differ when comparing
# a quantized rollout (Qwen3-32B-FP8) against a bf16 reference of the same
# checkpoint (Qwen3-32B) — both rows share MODEL_ID=Qwen/Qwen3-32B and
# differ only on engine_label.
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-32B")
ENGINE_LABEL = os.environ.get("ENGINE_LABEL", "vllm-bf16")
ENGINE_FINGERPRINT = os.environ.get(
    "ENGINE_FINGERPRINT", f"sha256:{ENGINE_LABEL}-tp4-l40s"
)
ENGINE_VERSION = os.environ.get("ENGINE_VERSION", "0.20.1")
TASKS_FILE = os.environ.get("TASKS_FILE", "scripts/live/agent_tasks.json")
OUT_DIR = Path(os.environ.get("OUT_DIR", f"runs/live/agent/{ENGINE_LABEL}"))
SEED = int(os.environ.get("SEED", "1234"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "8"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))


SYSTEM_PROMPT = (
    "You are a careful research assistant with access to tools. "
    "When the user asks for a calculation, ALWAYS call the calculator tool "
    "instead of doing the arithmetic yourself. When they ask for facts, "
    "use web_search. When they reference a file path, use read_file. "
    "After tool calls return, provide a concise final answer as plain text. "
    "Do not call tools forever — answer once you have enough information."
)

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for a query and return a short summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a Python arithmetic expression and return the result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python expression"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_eval",
            "description": "Evaluate a Python expression in a sandbox; returns its repr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Python expression"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its content. The file content is simulated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                },
                "required": ["path"],
            },
        },
    },
]


# ---- deterministic simulated tools -----------------------------------------

_SIM_FILES: dict[str, str] = {
    "/tmp/factorial.py": (
        "def factorial(n):\n"
        "    if n == 0:\n"
        "        return 0  # bug: should return 1\n"
        "    return n * factorial(n - 1)\n"
    ),
}


_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(expr: str) -> str:
    """Evaluate an arithmetic expression without exposing builtins."""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError as exc:
        return f"SyntaxError: {exc}"

    def _eval(n: ast.AST) -> float | int:
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)):
                return n.value
            raise TypeError(f"unsupported constant {n.value!r}")
        if isinstance(n, ast.BinOp):
            return _SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        if isinstance(n, ast.UnaryOp):
            return _SAFE_OPS[type(n.op)](_eval(n.operand))
        raise TypeError(f"unsupported expression node {type(n).__name__}")

    try:
        result = _eval(node)
    except Exception as exc:
        return f"Error: {exc}"
    return repr(result)


def _simulated_search(query: str) -> str:
    """Deterministic simulated search results — same query → same answer."""
    q = query.lower()
    if "vllm" in q and ("release" in q or "version" in q or "latest" in q):
        return (
            "vLLM v0.20.1 (Aug 2025): adds CUDA-graph memory profiler, NIXL "
            "transport for KV cache offload, finegrained FP8 attention "
            "kernels for SM90+ GPUs, and improved tensor-parallel scheduling."
        )
    if "tokyo" in q and "population" in q:
        return "The population of Tokyo (Tokyo Metropolis) is approximately 13,960,000."
    if "osaka" in q and "population" in q:
        return "The population of Osaka (Osaka Prefecture) is approximately 2,750,000."
    if "sglang" in q:
        return (
            "sglang is an open-source LLM serving framework with a focus on "
            "structured generation, RadixAttention KV-cache reuse, and "
            "OpenAI-compatible HTTP serving."
        )
    if "vllm" in q:
        return (
            "vLLM is an open-source LLM inference and serving engine using "
            "paged attention, continuous batching, and tensor parallelism."
        )
    digest = hashlib.sha256(query.encode()).hexdigest()[:8]
    return f"[simulated search; no canned result for query={query!r} (digest={digest})]"


def execute_tool(name: str, arguments: dict) -> str:
    if name == "web_search":
        return _simulated_search(str(arguments.get("query", "")))
    if name == "calculator" or name == "python_eval":
        return _safe_eval(str(arguments.get("expression", "")))
    if name == "read_file":
        path = str(arguments.get("path", ""))
        return _SIM_FILES.get(path, f"[simulated; no canned content for {path!r}]")
    return f"[unknown tool {name!r}]"


# ---- agent loop ------------------------------------------------------------


def _try_parse_arguments(raw: str | dict) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return {"_raw": str(raw)}


def run_agent(client: OpenAI, task: dict) -> AgentTrajectory:
    """Run one agent task to completion, returning the captured trajectory."""
    task_id = task["task_id"]
    task_text = task["task_text"]
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_text},
    ]
    steps: list[AgentStep] = []
    final_answer = ""
    terminal = "max_steps"
    total_assistant_tokens = 0

    for step_idx in range(MAX_STEPS):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.7,
                top_p=0.95,
                seed=SEED,
                max_tokens=MAX_TOKENS,
                # Qwen3 supports a "thinking" mode that interleaves chain-of-
                # thought tokens before the actual answer. Disable it so the
                # token budget is spent on tool calls + concise answers.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        except Exception as exc:
            steps.append(AgentStep(
                step_idx=step_idx,
                role="assistant",
                content=f"[client error: {exc}]",
            ))
            terminal = "error"
            break

        msg = resp.choices[0].message
        finish_reason = resp.choices[0].finish_reason
        content = msg.content or ""
        tool_calls_raw = msg.tool_calls or []
        usage = getattr(resp, "usage", None)
        if usage and getattr(usage, "completion_tokens", None) is not None:
            total_assistant_tokens += int(usage.completion_tokens)

        # Append assistant step.
        captured_calls: list[ToolCallRecord] = []
        for call in tool_calls_raw:
            args = _try_parse_arguments(call.function.arguments)
            result = execute_tool(call.function.name, args)
            captured_calls.append(
                ToolCallRecord.from_call(
                    name=call.function.name, arguments=args, result=result
                )
            )

        steps.append(AgentStep(
            step_idx=step_idx,
            role="assistant",
            content=content,
            tool_calls=captured_calls,
            finish_reason=finish_reason,
            response_token_count=int(getattr(usage, "completion_tokens", 0) or 0),
        ))

        if not tool_calls_raw:
            final_answer = content
            terminal = "answered"
            break

        # Echo the assistant turn back into messages for the next request.
        # We rebuild a minimal dict because openai client returns objects.
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls_raw
            ],
        })

        # Execute tools and append tool-role steps + messages.
        for call, captured in zip(tool_calls_raw, captured_calls):
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": captured.result_text,
            })
            steps.append(AgentStep(
                step_idx=step_idx,
                role="tool",
                content=captured.result_text,
            ))
    else:
        terminal = "max_steps"

    return AgentTrajectory(
        run_id=f"{ENGINE_LABEL}-{task_id}",
        task_id=task_id,
        task_text=task_text,
        model_id=MODEL_ID,
        rollout_engine=EngineFingerprint(
            name=ENGINE_LABEL, version=ENGINE_VERSION, fingerprint=ENGINE_FINGERPRINT
        ),
        available_tools=[t["function"]["name"] for t in TOOL_SCHEMAS],
        steps=steps,
        success=None,
        terminal_reason=terminal,
        total_assistant_tokens=total_assistant_tokens,
        final_answer=final_answer,
        created_at=datetime.now(timezone.utc),
    )


def main() -> int:
    tasks = json.loads(Path(TASKS_FILE).read_text())
    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        task_id = task["task_id"]
        out_file = OUT_DIR / f"{task_id}.json"
        if out_file.exists():
            print(f"[skip] {task_id}: already exists at {out_file}", flush=True)
            continue
        t0 = time.time()
        traj = run_agent(client, task)
        out_file.write_text(traj.model_dump_json(indent=2))
        dt = time.time() - t0
        n_assist = len(traj.assistant_steps())
        print(
            f"[done] {task_id}: {n_assist} assistant steps, "
            f"terminal={traj.terminal_reason}, {dt:.1f}s -> {out_file}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
