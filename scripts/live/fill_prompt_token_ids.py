"""Fill in missing prompt_token_ids on proxy-captured AgentTrajectories.

The logprob-capture proxy observes the chat-completions response and
emits ``response_logprobs`` + ``response_token_ids`` per turn, but the
OpenAI chat-completions response shape does **not** expose the prompt
token IDs vLLM consumed. ``build_hermes_dense_input.py`` requires both
sides to be present (the trainer-side teacher-force needs the exact
``prompt_token_ids`` to reproduce the rollout context), so this script
bridges the gap: for every assistant step that has
``response_token_ids`` populated but ``prompt_token_ids`` is ``None``,
it re-derives the prompt via the model's chat template (using the
trajectory's own accumulated history) and stamps the result onto the
step.

The captured ``response_token_ids`` are preserved verbatim — only the
prompt side is re-derived, since vLLM's prompt encoding is
chat-template-deterministic given the message history. This is the
realistic "partial capture" path already exercised by
``extract_token_pairs`` and matches what
``adapt_hermes_sharegpt_trajectory`` produces from proxy-captured
trajectories.

Trajectories are loaded via ``load_trajectory`` (Pydantic
``model_validate_json``) and written back via ``model_dump_json``,
preserving every other field unchanged. Trajectories that have NO
turns needing fill are left untouched on disk.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from rollout_market.observatory.agent_trajectory_lab import (
    AgentTrajectory,
    load_trajectory,
)


@runtime_checkable
class ChatTokenizer(Protocol):
    def apply_chat_template(
        self,
        conversation: list[dict],
        *,
        add_generation_prompt: bool = ...,
        tokenize: bool = ...,
    ) -> object: ...


def _load_extractor():
    """Lazy-import hermes_token_extraction.py for its prompt builders."""
    spec_path = Path(__file__).resolve().parent / "hermes_token_extraction.py"
    spec = importlib.util.spec_from_file_location("hermes_token_extraction_for_fill", spec_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fill_trajectory(
    trajectory: AgentTrajectory,
    tokenizer: ChatTokenizer,
    *,
    system_prompt: str | None = None,
) -> tuple[AgentTrajectory, int]:
    """Return ``(filled_trajectory, n_turns_filled)``.

    Pure function. For each assistant step whose
    ``response_token_ids`` is populated and ``prompt_token_ids`` is
    ``None``, the prompt is re-derived via chat-template encoding of
    the trajectory's accumulated history up to (but not including)
    that turn. Steps where the prompt is already present, OR where the
    response side is absent, are passed through unchanged.

    The trajectory's other fields (engine, model_id, terminal_reason,
    final_answer, etc.) are preserved.
    """
    extractor = _load_extractor()
    new_steps: list = list(trajectory.steps)
    assistant_positions: list[int] = [
        idx for idx, s in enumerate(trajectory.steps) if s.role == "assistant"
    ]

    n_filled = 0
    for assistant_idx, step_idx in enumerate(assistant_positions):
        step = new_steps[step_idx]
        if step.response_token_ids is None:
            continue
        if step.prompt_token_ids is not None:
            continue
        prior_msgs = extractor._build_prior_messages(
            trajectory,
            target_assistant_idx=assistant_idx,
            system_prompt=system_prompt,
        )
        prompt_ids = extractor._coerce_to_id_list(
            tokenizer.apply_chat_template(
                prior_msgs,
                add_generation_prompt=True,
                tokenize=True,
            )
        )
        new_steps[step_idx] = step.model_copy(update={"prompt_token_ids": prompt_ids})
        n_filled += 1
    if n_filled == 0:
        return trajectory, 0
    return trajectory.model_copy(update={"steps": new_steps}), n_filled


def _load_tokenizer(model_id: str) -> ChatTokenizer:
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    return AutoTokenizer.from_pretrained(model_id)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fill prompt_token_ids on proxy-captured trajectories so "
            "build_hermes_dense_input.py can flatten them into the "
            "trainer-side rollouts payload."
        )
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory of AgentTrajectory JSONs to fill in place.",
    )
    parser.add_argument(
        "--model-id-override",
        default=None,
        help="Override the HF model id for tokenizer load (defaults to "
        "the first trajectory's model_id).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system prompt to prepend to every reconstructed history.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute fill counts without writing trajectories back.",
    )
    args = parser.parse_args(argv)

    in_dir = Path(args.trajectory_dir)
    if not in_dir.is_dir():
        print(f"[fatal] trajectory dir not found: {in_dir}", file=sys.stderr)
        return 2
    files = sorted(in_dir.glob("*.json"))
    if not files:
        print(f"[fatal] no trajectory JSON files in {in_dir}", file=sys.stderr)
        return 2

    tokenizer: ChatTokenizer | None = None
    cached_model_id: str | None = args.model_id_override
    total_filled = 0
    total_files_touched = 0
    for path in files:
        traj = load_trajectory(path)
        if tokenizer is None:
            cached_model_id = cached_model_id or traj.model_id
            tokenizer = _load_tokenizer(cached_model_id)
        filled, n = fill_trajectory(traj, tokenizer, system_prompt=args.system_prompt)
        if n > 0:
            total_filled += n
            total_files_touched += 1
            if not args.dry_run:
                path.write_text(filled.model_dump_json(indent=2), encoding="utf-8")
        verb = "would fill" if args.dry_run else "filled"
        print(f"[fill-prompt-ids] {path.name}: {verb} {n} turn(s)", flush=True)
    print(
        f"[fill-prompt-ids] {total_filled} turn(s) across {total_files_touched} "
        f"file(s){' (dry-run, no writes)' if args.dry_run else ''}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
