"""MoE counterpart of pair_hermes_dense_reports.py.

Cycle 3 v2 (probe-at-rollout). Pairs:

  * the rollout-side ``response_routed_experts`` captured on each
    assistant ``AgentStep`` (from the logprob-capture proxy with
    ``return_routed_experts=True``), and
  * the trainer-side ``expert_ids`` produced by
    ``run_fsdp_moe_reference.py`` / ``run_megatron_moe_reference.py``
    when fed the captured ``(prompt_token_ids, response_token_ids)``
    pairs,

into one ``RouterMismatchReport`` per ``(trajectory, assistant_turn)``
under ``runs/live/router/hermes-qwen3-30b-a3b-{bf16,fp8}-vs-{fsdp,
megatron}-bf16/``.

Trainer input shape (a list, indexed by ``prompt_idx`` matching the
index file from ``build_hermes_dense_input.py``)::

    [
      {
        "prompt_idx": 0,
        "model": "Qwen/Qwen3-30B-A3B",
        "engine": "fsdp",
        "num_layers": 48,
        "num_experts": 128,
        "top_k": 8,
        "expert_ids": [[[...]]]  // shape [token][layer][top_k]
      },
      ...
    ]

This matches the schema each per-prompt trainer-MoE script emits
today; an aggregator that batches multiple prompts into one JSON is
a sibling of ``build_hermes_dense_input.py``.

``pair_reports`` is a pure function — same testing pattern as the
dense pair script.
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
from rollout_market.observatory.dense_mismatch_lab import EngineFingerprint
from rollout_market.observatory.router_mismatch_lab import (
    RouterMismatchReport,
    RouterTrace,
    compute_router_mismatch,
)


_DEFAULT_TRAINER_ENGINE_VERSION = "torch-2.4"


def _trainer_engine_from_entry(entry: dict) -> EngineFingerprint:
    """Build the trainer-side ``EngineFingerprint`` from one trainer
    payload entry. Mirrors the dense pair script's helper but
    defaults are MoE-specific (``run_fsdp_moe_reference.py``)."""
    engine_name = entry.get("engine", "fsdp")
    fp = entry.get("engine_fingerprint", f"sha256:{engine_name}-moe-{entry.get('world_size', '?')}")
    return EngineFingerprint(
        name=engine_name,
        version=_DEFAULT_TRAINER_ENGINE_VERSION,
        fingerprint=fp,
    )


def pair_reports(
    *,
    index: dict,
    trainers: list[dict],
    trajectories_by_run_id: dict[str, AgentTrajectory],
    precision_class: str,
    trainer_engine_label: str | None = None,
) -> list[RouterMismatchReport]:
    """Join index + trainer router traces into one report per turn.

    Pure function. Raises ``ValueError`` on:
      * missing ``prompt_idx`` in the trainer list;
      * missing ``response_routed_experts`` on the matching AgentStep;
      * shape mismatch between rollout and trainer traces (number of
        tokens / layers / top_k / experts);
      * trajectory_run_id from the index not found in
        ``trajectories_by_run_id``.
    """
    entries: list[dict] = index["entries"]
    trainers_by_idx: dict[int, dict] = {t["prompt_idx"]: t for t in trainers}

    reports: list[RouterMismatchReport] = []
    for entry in entries:
        prompt_idx = entry["prompt_idx"]
        if prompt_idx not in trainers_by_idx:
            raise ValueError(
                f"trainer entry missing for prompt_idx={prompt_idx} "
                f"(task_id={entry['task_id']}, turn={entry['assistant_turn_idx']})"
            )
        traj = trajectories_by_run_id.get(entry["trajectory_run_id"])
        if traj is None:
            raise ValueError(
                f"trajectory_run_id={entry['trajectory_run_id']!r} not found "
                "among supplied trajectories"
            )
        assistant_idx = entry["assistant_turn_idx"]
        assistants = [s for s in traj.steps if s.role == "assistant"]
        if assistant_idx >= len(assistants):
            raise ValueError(
                f"assistant_turn_idx={assistant_idx} out of range for "
                f"trajectory {entry['trajectory_run_id']!r} "
                f"(has {len(assistants)} assistant turns)"
            )
        step = assistants[assistant_idx]
        rollout_routed = step.response_routed_experts
        if rollout_routed is None:
            raise ValueError(
                f"prompt_idx={prompt_idx}: AgentStep has no "
                "response_routed_experts captured at rollout time"
            )

        t_entry = trainers_by_idx[prompt_idx]
        trainer_expert_ids = t_entry["expert_ids"]
        # The trainer-side trace covers all `prompt_len + response_len - 1`
        # predicting positions (see `routed_position_indices`). The rollout
        # sidecar is trimmed by the capture proxy to the response-token
        # window only (the engine emits prompt+gen-1 entries; the proxy
        # keeps the last `completion_tokens`). Slice the trainer trace to
        # the same response window so the per-token, per-layer flip rate
        # is computed on the same positions.
        rollout_len = len(rollout_routed)
        if len(trainer_expert_ids) > rollout_len:
            trainer_expert_ids = trainer_expert_ids[-rollout_len:]
        num_layers = int(t_entry["num_layers"])
        num_experts = int(t_entry["num_experts"])
        top_k = int(t_entry["top_k"])
        rollout_trace = RouterTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            expert_ids=rollout_routed,
        )
        trainer_trace = RouterTrace(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            expert_ids=trainer_expert_ids,
        )

        rollout_engine = traj.rollout_engine
        trainer_engine = _trainer_engine_from_entry(t_entry)
        if trainer_engine_label is not None:
            trainer_engine = EngineFingerprint(
                name=trainer_engine_label,
                version=trainer_engine.version,
                fingerprint=trainer_engine.fingerprint,
            )

        prompt_id = f"{entry['task_id']}-turn{assistant_idx}"
        run_id = f"{rollout_engine.name}-vs-{trainer_engine.name}-{prompt_id}"
        report = compute_router_mismatch(
            run_id=run_id,
            model_id=traj.model_id,
            rollout_engine=rollout_engine,
            trainer_engine=trainer_engine,
            rollout_trace=rollout_trace,
            trainer_trace=trainer_trace,
            notes=[
                f"hermes-agent moe pair: trajectory_run_id="
                f"{entry['trajectory_run_id']!r}, "
                f"assistant_turn_idx={assistant_idx}, "
                f"precision={precision_class}"
            ],
        )
        reports.append(report)
    return reports


def write_reports(
    reports: list[RouterMismatchReport],
    out_root: Path,
) -> list[Path]:
    """Write one ``router_mismatch_report.json`` per report under
    ``out_root``. Mirrors ``pair_hermes_dense_reports.write_reports``.

    The per-report subdirectory uses the ``run_id`` so the dashboard
    renderer's glob picks it up by-tile.
    """
    written: list[Path] = []
    out_root.mkdir(parents=True, exist_ok=True)
    for r in reports:
        sub = out_root / r.run_id
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / "router_mismatch_report.json"
        path.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        written.append(path)
    return written


def _load_trajectories(traj_dir: Path) -> dict[str, AgentTrajectory]:
    out: dict[str, AgentTrajectory] = {}
    for p in sorted(traj_dir.glob("*.json")):
        traj = load_trajectory(p)
        out[traj.run_id] = traj
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pair Hermes-Agent MoE rollout probes against trainer-side "
            "expert_ids; emit one RouterMismatchReport per (trajectory, "
            "assistant_turn)."
        )
    )
    parser.add_argument("--index", required=True)
    parser.add_argument(
        "--trainers",
        required=True,
        help="Trainer-side payload list (one entry per prompt_idx).",
    )
    parser.add_argument("--trajectory-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument(
        "--trainer-engine-label",
        default=None,
        help="Override the trainer engine's name (e.g. fsdp-bf16, megatron-lm).",
    )
    args = parser.parse_args(argv)

    index_path = Path(args.index)
    trainers_path = Path(args.trainers)
    traj_dir = Path(args.trajectory_dir)
    out_root = Path(args.out_root)
    for p in (index_path, trainers_path):
        if not p.exists():
            print(f"[fatal] not found: {p}", file=sys.stderr)
            return 2
    if not traj_dir.is_dir():
        print(f"[fatal] trajectory dir not found: {traj_dir}", file=sys.stderr)
        return 2

    index = json.loads(index_path.read_text(encoding="utf-8"))
    trainers = json.loads(trainers_path.read_text(encoding="utf-8"))
    trajectories = _load_trajectories(traj_dir)
    reports = pair_reports(
        index=index,
        trainers=trainers,
        trajectories_by_run_id=trajectories,
        precision_class=index["precision_class"],
        trainer_engine_label=args.trainer_engine_label,
    )
    written = write_reports(reports, out_root)
    print(
        f"[hermes-moe-pair] {len(reports)} report(s) written under {out_root}",
        flush=True,
    )
    for p in written:
        print(f"  - {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
