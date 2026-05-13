"""Tool-using OpenAI driver with client-side probe capture.

Sibling of ``minimal_agent_driver.py``. The difference: instead of
relying on a co-located ``logprob_capture_proxy.py`` to rewrite each
request and skim each response into a sidecar JSONL, this driver
does the rewrite + skim *itself*. That moves the proxy's role to the
coordinator host (this script) and lets a remote operator serve
vLLM with no extra processes inside their environment.

The wire shape and downstream consumers are unchanged: the sidecar
JSONL this driver writes is byte-equivalent (modulo timestamps /
request_ids) to what the proxy emits, so
``merge_probes_into_trajectories.py`` works the same way.

Run from the laptop (or any coordinator host):

    python scripts/live/remote_capture_driver.py \\
        --base-url http://localhost:8001 \\        # SSH-forwarded
        --model Qwen/Qwen3-30B-A3B \\
        --tasks scripts/live/agent_tasks_hermes.json \\
        --task-id math-tokens-per-gpu-day \\
        --out /tmp/minimal_sharegpt_remote.jsonl \\
        --sidecar /tmp/hermes_probes_remote.jsonl \\
        --max-steps 6 --max-tokens 1024 --timeout 240
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# scripts/live/ is not a package — make sibling helpers importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests  # type: ignore[import-not-found]

# Reuse the probe-extraction contract from the proxy verbatim — same
# request augmentation, same response-side schema. Importing them
# guarantees the downstream merger script sees identical records.
from logprob_capture_proxy import (  # type: ignore[import-not-found]
    derive_turn_idx,
    extract_probes,
    inject_logprobs,
)

# Reuse the rest of the minimal driver: tool dispatch, task loop,
# ShareGPT serialization, CLI arg shape. We only override
# ``_chat_once`` to add the client-side probe capture.
from minimal_agent_driver import (  # type: ignore[import-not-found]
    TOOLS_OPENAI_SPEC,
    _conv_from_messages,
    _dispatch_tool,
)


def _chat_once_with_capture(
    *,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: float,
    sidecar_path: Path,
    prompt_index: int | str,
    tool_choice: str = "auto",
) -> dict:
    """POST one chat-completions request, capture probes client-side.

    Mirrors ``minimal_agent_driver._chat_once`` but:

    * augments the outgoing body with ``inject_logprobs`` so the
      response carries ``logprobs.content`` / ``prompt_token_ids`` /
      ``choices[i].token_ids`` / ``choices[i].routed_experts``;
    * captures the raw request + response bytes after the POST;
    * calls ``extract_probes`` to build the sidecar record and
      appends one line to ``sidecar_path`` (the same format
      ``logprob_capture_proxy.py`` writes).

    Returns the parsed JSON response dict so the caller's tool loop
    works unchanged.
    """
    body = {
        "model": model,
        "messages": messages,
        "tools": TOOLS_OPENAI_SPEC,
        "tool_choice": tool_choice,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        # Don't set chat_template_kwargs here; inject_logprobs sets it.
    }
    raw_request = json.dumps(body).encode("utf-8")
    augmented = inject_logprobs(raw_request)
    r = requests.post(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=augmented,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    raw_response = r.content

    turn_idx = derive_turn_idx(augmented)
    record = extract_probes(
        augmented,
        raw_response,
        prompt_index=prompt_index,
        turn_idx=turn_idx,
    )
    if record is not None:
        record["timestamp"] = time.time()
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")

    return r.json()


def _run_one_task_with_capture(
    task: dict,
    *,
    base_url: str,
    model: str,
    max_steps: int,
    max_tokens: int,
    timeout: float,
    sidecar_path: Path,
) -> dict:
    """Tool-using task loop. Identical to
    ``minimal_agent_driver._run_one_task`` except every chat call
    flows through ``_chat_once_with_capture`` so each assistant turn
    gets one line in the sidecar JSONL.
    """
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
    prompt_index = task.get("prompt_index", 0)
    for _ in range(max_steps):
        try:
            rsp = _chat_once_with_capture(
                base_url=base_url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                timeout=timeout,
                sidecar_path=sidecar_path,
                prompt_index=prompt_index,
            )
        except requests.RequestException as exc:
            print(
                f"[remote-capture] {task['task_id']}: chat error: {exc}",
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
                args = (
                    json.loads(fn["arguments"])
                    if isinstance(fn["arguments"], str)
                    else fn["arguments"]
                )
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
        "prompt_index": prompt_index,
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
            "Tool-using OpenAI driver with client-side probe capture. "
            "Same trajectory shape as minimal_agent_driver.py but writes "
            "the probe sidecar JSONL itself instead of routing through "
            "logprob_capture_proxy.py — for the remote-vLLM case where "
            "the operator can't run the proxy inside their environment."
        )
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True, help="ShareGPT JSONL output path")
    parser.add_argument(
        "--sidecar",
        default="/tmp/hermes_probes.jsonl",
        help=(
            "Probe sidecar JSONL output. Defaults to the same path "
            "logprob_capture_proxy.py uses so downstream merge scripts "
            "work without changes."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--task-id",
        default=None,
        help="Drive only this task_id from --tasks (default: all).",
    )
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args(argv)

    tasks_path = Path(args.tasks)
    if not tasks_path.is_file():
        print(f"[remote-capture] tasks file not found: {tasks_path}", file=sys.stderr)
        return 2
    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    if args.task_id is not None:
        tasks = [t for t in tasks if t.get("task_id") == args.task_id]
        if not tasks:
            print(f"[remote-capture] no task with id {args.task_id!r}", file=sys.stderr)
            return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(args.sidecar)
    if sidecar.exists():
        # The proxy mode appends as well; we follow suit so multiple
        # runs accumulate into one sidecar.
        pass

    with out_path.open("w", encoding="utf-8") as fh:
        for i, task in enumerate(tasks):
            task["prompt_index"] = i
            print(
                f"[remote-capture] starting {task['task_id']} (prompt_index={i})",
                flush=True,
            )
            t0 = time.time()
            record = _run_one_task_with_capture(
                task,
                base_url=args.base_url,
                model=args.model,
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                sidecar_path=sidecar,
            )
            elapsed = time.time() - t0
            terminal = (
                "answered"
                if record["completed"]
                else ("partial" if record["partial"] else "max_steps")
            )
            print(
                f"[remote-capture]   {task['task_id']}: terminal={terminal} "
                f"api_calls={record['api_calls']} {elapsed:.1f}s",
                flush=True,
            )
            fh.write(json.dumps(record) + "\n")

    print(f"[remote-capture] wrote {out_path}", flush=True)
    print(f"[remote-capture] probes sidecar: {sidecar}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
