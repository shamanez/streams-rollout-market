"""Minimal multi-turn agent driver that talks to a vLLM endpoint via
the OpenAI Chat Completions API and produces:

  * one hermes-agent-shaped ShareGPT JSONL (the merger consumes this);
  * a sidecar probe file populated by the proxy in the same flow.

Architecture (cycle 3 v2, probe-at-rollout):

    minimal_agent_driver.py  ──┐
                               ├─→  logprob_capture_proxy.py
                               │     (rewrites the request, captures
                               │      sidecar probes per assistant turn)
                               └─→  vLLM serve on the spot
                                    (--enable-return-routed-experts for MoE)

The driver speaks plain OpenAI chat-completions with tool calls and
loops turn-by-turn until either (a) the model produces no further
tool calls (terminal "answered") or (b) MAX_STEPS turns have elapsed
(terminal "max_steps").

Tools supported (kept deliberately tiny — same set the cycle-2 agent
runner exposed and the same set the cycle-3 v2 task list mentions):

    calculator       — eval a Python arithmetic expression
    python_eval      — eval a Python expression
    read_file        — read the contents of a local file
    web_search       — return a canned stub result string

Everything bigger than these belongs in the real Hermes Agent or a
follow-on driver. The point here is to produce trajectories the
proxy can probe at rollout time so the dashboard cells finally fill
in.
"""

from __future__ import annotations

import argparse
import ast
import json
import operator
import os
import sys
import time
from pathlib import Path

try:
    import requests  # type: ignore[import-not-found]
except ImportError as exc:  # pragma: no cover - runtime guard only
    raise SystemExit(
        "[minimal-agent] requests is required: pip install requests"
    ) from exc


_MAX_STEPS_DEFAULT = 25
_DEFAULT_MAX_TOKENS = 512


# --- tool runtime ------------------------------------------------------------

# Calculator: safe subset of Python AST evaluation (no name lookups,
# no attribute access, no calls). Mirrors the calculator tool every
# cycle-2 agent run exposed.
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str) -> str:
    """Pure-arithmetic evaluator for the calculator + python_eval tools."""

    def visit(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](visit(node.operand))
        raise ValueError(f"unsupported syntax: {ast.dump(node)}")

    try:
        tree = ast.parse(expr, mode="eval")
        return str(visit(tree))
    except (ValueError, SyntaxError, ZeroDivisionError) as exc:
        return f"<error: {exc}>"


def _read_file(path: str) -> str:
    """Read a small text file off the host filesystem."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            return fh.read(8192)
    except OSError as exc:
        return f"<error: {exc}>"


# Stable stub results — the trajectory's value is in the agent's
# tool-call decisions, not in real-time web data.
_WEB_STUB: dict[str, str] = {
    "vllm": "vllm is an open-source high-throughput LLM serving framework.",
    "sglang": "sglang is a serving framework with a structured generation language.",
    "tokyo population": "Tokyo: ~37 million metropolitan population.",
    "osaka population": "Osaka: ~19 million metropolitan population.",
}


def _web_search(query: str) -> str:
    q = (query or "").strip().lower()
    return _WEB_STUB.get(
        q, f"<stub: no canned result for query={q!r}; returning empty>"
    )


TOOLS_OPENAI_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a Python arithmetic expression and return the result as a string.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_eval",
            "description": "Evaluate a pure Python arithmetic expression (no imports, no I/O).",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read up to 8 KiB of text from a local file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Return a canned stub result for a small set of known queries.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def _dispatch_tool(name: str, args: dict) -> str:
    if name in ("calculator", "python_eval"):
        return _safe_eval(args.get("expression", ""))
    if name == "read_file":
        return _read_file(args.get("path", ""))
    if name == "web_search":
        return _web_search(args.get("query", ""))
    return f"<error: unknown tool {name!r}>"


# --- ShareGPT serialisation --------------------------------------------------


def _conv_from_messages(messages: list[dict]) -> list[dict]:
    """Convert an OpenAI-shape ``messages`` list into the ShareGPT
    ``from/value`` shape ``merge_probes_into_trajectories`` consumes."""
    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            out.append({"from": "system", "value": msg["content"]})
        elif role == "user":
            out.append({"from": "human", "value": msg["content"]})
        elif role == "assistant":
            # Inline tool_calls into a <tool_call>...</tool_call> block
            # so adapt_hermes_sharegpt_trajectory recognises them.
            parts: list[str] = []
            text = msg.get("content") or ""
            if text:
                parts.append(text)
            for call in msg.get("tool_calls") or []:
                fn = call["function"]
                payload = {"name": fn["name"], "arguments": json.loads(fn["arguments"])}
                parts.append(
                    "<tool_call>\n" + json.dumps(payload) + "\n</tool_call>"
                )
            out.append({"from": "gpt", "value": "\n".join(parts)})
        elif role == "tool":
            payload = {
                "tool_call_id": msg.get("tool_call_id", ""),
                "name": msg.get("name", "unknown"),
                "content": msg.get("content", ""),
            }
            out.append(
                {
                    "from": "tool",
                    "value": "<tool_response>\n"
                    + json.dumps(payload)
                    + "\n</tool_response>",
                }
            )
    return out


# --- chat-completions client -------------------------------------------------


def _chat_once(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    tool_choice: str = "auto",
) -> dict:
    """Single chat-completions POST. Returns the JSON response dict."""
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS_OPENAI_SPEC,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    r = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        json=body,
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def _run_one_task(
    task: dict,
    *,
    base_url: str,
    model: str,
    max_steps: int,
    max_tokens: int,
    timeout: float,
) -> dict:
    """Drive one task to completion (or max_steps); return a ShareGPT record."""
    system = (
        "You are a tool-using agent operating in a loop. The user will give you "
        "a task that requires tools. You MUST call tools in your first turn "
        "instead of solving from memory — keep your <think> block short and "
        "emit a tool_call as soon as you've named the tool to use. Continue "
        "calling tools across turns until you've gathered enough information; "
        "then give a final plain-text answer with NO further tool_calls. "
        "Do NOT reason your way to the final answer without using any tools."
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": task["task_text"]},
    ]
    api_calls = 0
    completed = False
    partial = False
    for _ in range(max_steps):
        try:
            rsp = _chat_once(
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            print(
                f"[minimal-agent] {task['task_id']}: chat error: {exc}",
                file=sys.stderr,
                flush=True,
            )
            partial = True
            break
        api_calls += 1
        choice = rsp["choices"][0]
        msg = choice["message"]
        assistant_msg: dict = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            assistant_msg["tool_calls"] = msg["tool_calls"]
        messages.append(assistant_msg)
        if not msg.get("tool_calls"):
            completed = True
            break
        for call in msg["tool_calls"]:
            fn = call["function"]
            try:
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            except json.JSONDecodeError:
                args = {"_raw": fn["arguments"]}
            result = _dispatch_tool(fn["name"], args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": fn["name"],
                    "content": result,
                }
            )

    return {
        "prompt_index": task.get("prompt_index", 0),
        "task_id": task["task_id"],
        "task_text": task["task_text"],
        "conversations": _conv_from_messages(messages),
        "completed": completed,
        "partial": partial,
        "api_calls": api_calls,
        "max_steps": max_steps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Drive a multi-turn agent loop against a vLLM OpenAI endpoint. "
            "Emits one ShareGPT JSONL line per task; pair with the "
            "logprob_capture_proxy sidecar via merge_probes_into_trajectories.py."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True, help="Path to agent_tasks_hermes.json")
    parser.add_argument("--out", required=True, help="ShareGPT JSONL output path")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("MAX_STEPS", _MAX_STEPS_DEFAULT)),
    )
    parser.add_argument("--max-tokens", type=int, default=_DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--task-id",
        default=None,
        help="If set, only run the task with this task_id.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"[fatal] tasks file not found: {tasks_path}", file=sys.stderr)
        return 2
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.task_id is not None:
        tasks = [t for t in tasks if t["task_id"] == args.task_id]
        if not tasks:
            print(f"[fatal] no task with task_id={args.task_id!r}", file=sys.stderr)
            return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks):
            task.setdefault("prompt_index", i)
            t0 = time.time()
            print(
                f"[minimal-agent] starting {task['task_id']} (prompt_index={i})",
                flush=True,
            )
            record = _run_one_task(
                task,
                base_url=args.base_url,
                model=args.model,
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            fh.write(json.dumps(record) + "\n")
            fh.flush()
            dt = time.time() - t0
            term = (
                "answered"
                if record["completed"]
                else ("error" if record["partial"] else "max_steps")
            )
            print(
                f"[minimal-agent]   {task['task_id']}: terminal={term} "
                f"api_calls={record['api_calls']} {dt:.1f}s",
                flush=True,
            )
    print(f"[minimal-agent] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
