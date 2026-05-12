"""Lift (prompt, response) token pairs out of Hermes-Agent trajectories.

Cycle-3 (STEER.md, 2026-05-12) re-feeds every Hermes-Agent response
token sequence through the trainer-side engines (FSDP-bf16,
Megatron-bf16) and computes the same token-level metrics already used
elsewhere — ESS for Dense, ``router_flip_rate`` for MoE — so the
``Hermes Agent`` matrices mirror the existing Dense + MoE
engine-vs-trainer matrices.

This module is the seam between the AgentTrajectory contract and the
``/tmp/rollout.json``-style payload that
``scripts/live/run_fsdp_reference.py``,
``scripts/live/run_fsdp_moe_reference.py``, and
``scripts/live/run_megatron_moe_reference.py`` consume. It walks an
AgentTrajectory, reconstructs the chat-templated context that preceded
each assistant turn, and emits ``(prompt_token_ids,
response_token_ids)`` — one entry per assistant turn.

The core function is pure: it takes an ``AgentTrajectory`` and a
chat-templating tokenizer and returns the token pairs. The CLI exists
so an operator can sweep a trajectory directory and dump
``/tmp/hermes_tokens_<task_id>.json`` payloads ready to be scp'd onto
the spot instance and fed to the existing trainer-reference scripts.

Honors STEER.md global directives:
  * one feature-results entry per commit
  * vLLM is rollout only; no sglang anywhere in the headline
  * the trajectory text is the marketplace artefact, the token-level
    re-feed measures whether different trainer engines AGREE on the
    per-token logprobs / router decisions for THOSE EXACT TOKENS
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from rollout_market.observatory.agent_trajectory_lab import (
    AgentTrajectory,
    load_trajectory,
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@runtime_checkable
class ChatTokenizer(Protocol):
    """Minimal protocol for a HuggingFace-style chat-templating tokenizer.

    Both ``AutoTokenizer.from_pretrained(...)`` and the test fixture in
    ``tests/test_hermes_token_extraction.py`` satisfy this shape.
    """

    def apply_chat_template(
        self,
        conversation: list[dict],
        *,
        add_generation_prompt: bool = ...,
        tokenize: bool = ...,
    ) -> list[int]: ...


def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks; keep ``<tool_call>`` bodies intact.

    Qwen3's chat template drops prior-turn chain-of-thought from history
    by default, and Hermes Agent's ShareGPT output preserves the
    ``<think>`` markers verbatim. Stripping them on both the history side
    and the response side mirrors what the model actually sees / emits.
    """
    return _THINK_RE.sub("", text).strip()


def _build_prior_messages(
    trajectory: AgentTrajectory,
    *,
    target_assistant_idx: int,
    system_prompt: str | None,
) -> list[dict]:
    """Reconstruct the chat history fed to the model *before* the
    ``target_assistant_idx``-th assistant turn was emitted.

    Walks the trajectory in order: prior assistant turns retain their
    Hermes-emitted content (``<think>`` stripped, ``<tool_call>`` XML
    retained); tool steps become ``role=tool`` messages with their raw
    content text. Stops as soon as the target assistant turn is reached.
    """
    msgs: list[dict] = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": trajectory.task_text})
    seen_assistants = 0
    for step in trajectory.steps:
        if step.role == "assistant":
            if seen_assistants >= target_assistant_idx:
                break
            msgs.append({"role": "assistant", "content": _strip_think(step.content)})
            seen_assistants += 1
        else:  # role == "tool"
            msgs.append({"role": "tool", "content": step.content})
    return msgs


def extract_token_pairs(
    trajectory: AgentTrajectory,
    tokenizer: ChatTokenizer,
    *,
    system_prompt: str | None = None,
) -> list[tuple[list[int], list[int]]]:
    """Return one ``(prompt_token_ids, response_token_ids)`` pair per
    assistant turn in ``trajectory``.

    For assistant turn ``i``:

      * **prompt** = chat-templated encoding of ``system + user +
        assistant_turns[0..i-1]`` interleaved with their tool responses,
        terminated by the assistant generation prompt
        (``add_generation_prompt=True``).
      * **response** = the encoding of the chat template applied to
        ``prior_messages + [assistant turn i]`` with
        ``add_generation_prompt=False``, sliced past the prompt prefix.
        ``<think>`` blocks are stripped from the response text;
        ``<tool_call>`` XML blocks are kept as Hermes emitted them.

    Pure function — no file I/O, no env reads. Raises ``ValueError`` if
    the chat template does not extend its output for the assistant turn
    (a contract violation, surfaced loudly so callers can investigate
    the tokenizer).
    """
    pairs: list[tuple[list[int], list[int]]] = []
    assistant_steps = [s for s in trajectory.steps if s.role == "assistant"]
    for i, step in enumerate(assistant_steps):
        prior_msgs = _build_prior_messages(
            trajectory,
            target_assistant_idx=i,
            system_prompt=system_prompt,
        )
        response_text = _strip_think(step.content)
        prompt_ids = list(
            tokenizer.apply_chat_template(
                prior_msgs,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        full_ids = list(
            tokenizer.apply_chat_template(
                prior_msgs + [{"role": "assistant", "content": response_text}],
                add_generation_prompt=False,
                tokenize=True,
            )
        )
        if len(full_ids) <= len(prompt_ids):
            raise ValueError(
                "chat template produced a non-extending response for "
                f"assistant turn {i}: prompt_len={len(prompt_ids)}, "
                f"full_len={len(full_ids)} — check tokenizer chat template"
            )
        response_ids = full_ids[len(prompt_ids) :]
        pairs.append((prompt_ids, response_ids))
    return pairs


def trajectory_to_rollouts(
    trajectory: AgentTrajectory,
    tokenizer: ChatTokenizer,
    *,
    system_prompt: str | None = None,
) -> list[dict]:
    """Render a trajectory's token pairs as the ``rollouts.json`` shape
    consumed by ``run_fsdp_reference.py`` /
    ``run_fsdp_moe_reference.py`` / ``run_megatron_moe_reference.py``.

    One entry per assistant turn. Each entry carries the load-bearing
    keys those scripts read (``prompt_token_ids``, ``response_token_ids``,
    ``model``, ``prompt_idx``) plus provenance fields (``task_id``,
    ``assistant_turn_idx``, ``trajectory_run_id``) so downstream reports
    can be back-referenced to the originating trajectory.
    """
    pairs = extract_token_pairs(trajectory, tokenizer, system_prompt=system_prompt)
    out: list[dict] = []
    for i, (prompt_ids, response_ids) in enumerate(pairs):
        out.append(
            {
                "task_id": trajectory.task_id,
                "assistant_turn_idx": i,
                "model": trajectory.model_id,
                "prompt_idx": i,
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "trajectory_run_id": trajectory.run_id,
            }
        )
    return out


# --- CLI --------------------------------------------------------------------


def _load_tokenizer(model_id: str) -> ChatTokenizer:
    """Resolve a HuggingFace tokenizer for the given ``model_id``.

    Imported lazily so the module remains importable without
    ``transformers`` installed (matters for the offline unit tests).
    """
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(model_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract (prompt, response) token pairs from Hermes-Agent "
            "trajectories into rollouts.json-shaped payloads."
        )
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory of AgentTrajectory JSON files to walk.",
    )
    parser.add_argument(
        "--out-dir",
        default="/tmp",
        help="Where to write hermes_tokens_<task_id>.json (default: /tmp).",
    )
    parser.add_argument(
        "--model-id-override",
        default=None,
        help="Override the HF model id used to load the tokenizer.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to prepend to every message history.",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.trajectory_dir)
    if not in_dir.is_dir():
        print(f"[fatal] trajectory dir not found: {in_dir}", file=sys.stderr)
        return 2
    trajectories = sorted(in_dir.glob("*.json"))
    if not trajectories:
        print(f"[fatal] no trajectory JSON files in {in_dir}", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: ChatTokenizer | None = None
    cached_model_id: str | None = args.model_id_override
    for tpath in trajectories:
        traj = load_trajectory(tpath)
        if tokenizer is None:
            cached_model_id = cached_model_id or traj.model_id
            tokenizer = _load_tokenizer(cached_model_id)
        rollouts = trajectory_to_rollouts(traj, tokenizer, system_prompt=args.system_prompt)
        out_path = out_dir / f"hermes_tokens_{traj.task_id}.json"
        out_path.write_text(json.dumps(rollouts, indent=2), encoding="utf-8")
        print(
            f"[hermes-tokens] {traj.task_id}: {len(rollouts)} turn(s) -> {out_path}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
