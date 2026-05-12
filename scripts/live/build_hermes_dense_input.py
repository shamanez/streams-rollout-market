"""Aggregate Hermes-Agent trajectories into the trainer's rollouts.json shape.

Cycle-3 v2 (probe-at-rollout) wires up the dense matrix's Hermes-Agent
cells. After ``hermes_agent_runner.py`` writes one ``AgentTrajectory``
JSON per task — each assistant turn carrying its captured
``prompt_token_ids``, ``response_token_ids`` and ``response_logprobs``
from the vLLM forward pass — this script flattens the per-turn data
into the ``/tmp/rollouts.json`` payload that
``scripts/live/run_fsdp_reference.py`` already understands.

The trainer-side script outputs ``/tmp/trainers.json`` indexed by
``prompt_idx`` (a sequential int per entry in the input). To pair each
``trainer_logprobs`` back to the originating ``(trajectory, turn)`` —
and to recover the rollout-side ``response_logprobs`` captured by vLLM
at generation time — this builder also writes an *index* file:

    {
      "rollouts_path": "/tmp/rollouts_hermes_<precision>.json",
      "model_id": "Qwen/Qwen3-32B",
      "precision_class": "bf16",
      "entries": [
        {
          "prompt_idx": 0,
          "task_id": "<id>",
          "trajectory_run_id": "<id>",
          "assistant_turn_idx": 0,
          "num_tokens": 12,
          "response_logprobs": [-0.1, ...],
        },
        ...
      ]
    }

The downstream pairing script (next iteration) joins the index + the
trainer's outputs by ``prompt_idx`` and emits one
``DenseMismatchReport`` per ``(trajectory, turn)``.

This script does **not** touch the spot instance and runs offline
against trajectory JSONs already pulled back to the laptop. STEER.md
cycle-3 v2 global directives apply: vLLM is rollout only, one
feature-results entry per commit, prefer captured token IDs over
chat-template re-encoding.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rollout_market.observatory.agent_trajectory_lab import (
    AgentTrajectory,
    load_trajectory,
)


def build_rollouts_and_index(
    trajectories: list[AgentTrajectory],
    *,
    precision_class: str,
    model_id_override: str | None = None,
) -> tuple[list[dict], dict]:
    """Flatten a list of trajectories into ``(rollouts_payload, index)``.

    Pure function — testable offline against synthetic ``AgentTrajectory``
    fixtures. Only assistant turns carrying both ``prompt_token_ids``
    *and* ``response_token_ids`` produce rollout entries; assistant
    turns missing either field are skipped and recorded in the index's
    ``skipped`` list with a brief reason. Tool-role steps are ignored
    by design (they don't generate tokens under the model).

    Each rollout entry mirrors the schema ``run_fsdp_reference.py``
    consumes::

        {
          "model": "<HF model id>",
          "prompt_idx": <int>,
          "prompt_token_ids": [...],
          "response_token_ids": [...],
          "trainer_reference": "<HF model id>",
          "task_id": ...,
          "assistant_turn_idx": ...,
          "trajectory_run_id": ...,
        }

    ``trainer_reference`` is set equal to ``model`` by default; the
    trainer script reads ``trainer_reference`` and falls back to
    ``model`` if absent (the same fall-through it does for the legacy
    single-prompt pipeline).
    """
    rollouts: list[dict] = []
    index_entries: list[dict] = []
    skipped: list[dict] = []
    next_prompt_idx = 0
    resolved_model_id: str | None = model_id_override

    for traj in trajectories:
        if resolved_model_id is None:
            resolved_model_id = traj.model_id
        # We allow trajectories from the same precision to vary in
        # task_id but they must agree on model_id (otherwise the
        # downstream trainer-side teacher-force would mix models).
        if traj.model_id != resolved_model_id and model_id_override is None:
            raise ValueError(
                f"trajectory {traj.run_id} has model_id={traj.model_id!r}, "
                f"expected {resolved_model_id!r} — pass --model-id-override "
                "to force a single model on the trainer side"
            )

        assistant_turns = [s for s in traj.steps if s.role == "assistant"]
        for turn_idx, step in enumerate(assistant_turns):
            prompt_ids = step.prompt_token_ids
            response_ids = step.response_token_ids
            response_lp = step.response_logprobs
            if prompt_ids is None or response_ids is None:
                skipped.append(
                    {
                        "task_id": traj.task_id,
                        "trajectory_run_id": traj.run_id,
                        "assistant_turn_idx": turn_idx,
                        "reason": "missing prompt_token_ids or response_token_ids",
                    }
                )
                continue
            if response_lp is None:
                # Probes were partially populated — token IDs but no
                # rollout-side logprobs. The trainer-side forward pass
                # still works but the pairing has nothing to compare
                # against, so we skip.
                skipped.append(
                    {
                        "task_id": traj.task_id,
                        "trajectory_run_id": traj.run_id,
                        "assistant_turn_idx": turn_idx,
                        "reason": "missing response_logprobs (no rollout-side comparison possible)",
                    }
                )
                continue

            rollouts.append(
                {
                    "model": resolved_model_id,
                    "trainer_reference": resolved_model_id,
                    "prompt_idx": next_prompt_idx,
                    "prompt_token_ids": list(prompt_ids),
                    "response_token_ids": list(response_ids),
                    "task_id": traj.task_id,
                    "assistant_turn_idx": turn_idx,
                    "trajectory_run_id": traj.run_id,
                }
            )
            index_entries.append(
                {
                    "prompt_idx": next_prompt_idx,
                    "task_id": traj.task_id,
                    "trajectory_run_id": traj.run_id,
                    "assistant_turn_idx": turn_idx,
                    "num_tokens": len(response_ids),
                    "response_logprobs": list(response_lp),
                }
            )
            next_prompt_idx += 1

    index = {
        "model_id": resolved_model_id,
        "precision_class": precision_class,
        "entries": index_entries,
        "skipped": skipped,
    }
    return rollouts, index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Flatten Hermes-Agent AgentTrajectory JSONs (with rollout-side "
            "probes) into the /tmp/rollouts.json shape consumed by "
            "run_fsdp_reference.py, plus an index of (prompt_idx -> "
            "task_id, turn_idx, response_logprobs)."
        )
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory of AgentTrajectory JSON files (one per task).",
    )
    parser.add_argument(
        "--precision-class",
        required=True,
        choices=("bf16", "fp8"),
        help="Rollout precision the captured probes were emitted under.",
    )
    parser.add_argument(
        "--out-rollouts",
        default=None,
        help="Path for the trainer-side rollouts payload "
        "(default: /tmp/rollouts_hermes_<precision>.json).",
    )
    parser.add_argument(
        "--out-index",
        default=None,
        help="Path for the pairing index (default: /tmp/hermes_dense_index_<precision>.json).",
    )
    parser.add_argument(
        "--model-id-override",
        default=None,
        help="Force a single model id for all trajectories (defaults to the "
        "first trajectory's model_id; if subsequent trajectories disagree "
        "without an override the build aborts).",
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

    trajectories = [load_trajectory(p) for p in files]
    rollouts, index = build_rollouts_and_index(
        trajectories,
        precision_class=args.precision_class,
        model_id_override=args.model_id_override,
    )

    out_rollouts = Path(args.out_rollouts or f"/tmp/rollouts_hermes_{args.precision_class}.json")
    out_index = Path(args.out_index or f"/tmp/hermes_dense_index_{args.precision_class}.json")
    out_rollouts.parent.mkdir(parents=True, exist_ok=True)
    out_index.parent.mkdir(parents=True, exist_ok=True)
    out_rollouts.write_text(json.dumps(rollouts, indent=2), encoding="utf-8")
    index["rollouts_path"] = str(out_rollouts)
    out_index.write_text(json.dumps(index, indent=2), encoding="utf-8")

    n_skipped = len(index["skipped"])
    print(
        f"[hermes-dense] {len(trajectories)} trajectories -> "
        f"{len(rollouts)} probe-bearing turns ({n_skipped} skipped) -> "
        f"{out_rollouts} + {out_index}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
