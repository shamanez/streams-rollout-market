"""Pair Hermes-Agent rollout probes against trainer-side teacher-force.

The last setup step in the dense matrix's Hermes-Agent cell. Given:

  * an index file produced by ``build_hermes_dense_input.py`` (the
    ``prompt_idx -> (task_id, turn_idx, response_logprobs)`` mapping
    plus the rollout-side ``response_logprobs`` captured by vLLM at
    generation time);
  * the trainer-side output ``/tmp/trainers.json`` produced by
    ``run_fsdp_reference.py`` (a flat list keyed on the same
    ``prompt_idx``);
  * the originating directory of ``AgentTrajectory`` JSONs (used to
    pull engine fingerprints, ``model_id``, ``tokenizer_hash``, and a
    per-trajectory checkpoint_digest stand-in),

this script emits one ``DenseMismatchReport`` per ``(trajectory,
assistant_turn)`` under ``runs/live/dense/<pair-slug>/<task_id>-turn<k>/``.

The downstream dashboard renderer (``dense_dashboard.py``) ingests
these reports unchanged — the pair-slug becomes the
``(rollout_engine, trainer_engine, precision)`` tile for the Hermes
Agent dense matrix.

Pure logic lives in ``pair_reports`` so the joining + report
construction is testable offline without a trainer or spot dependency.
"""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

from rollout_market.observatory.agent_trajectory_lab import (
    AgentTrajectory,
    load_trajectory,
)
from rollout_market.observatory.dense_mismatch_lab import (
    DenseMismatchReport,
    EngineFingerprint,
    compute_dense_mismatch,
)


_DEFAULT_TRAINER_ENGINE_NAME = "fsdp"
_DEFAULT_TRAINER_ENGINE_VERSION = "torch-2.4"


def _trainer_engine_from_entry(entry: dict) -> EngineFingerprint:
    """Build an ``EngineFingerprint`` from one ``/tmp/trainers.json`` entry.

    Prefers the trainer's own ``engine_fingerprint`` when supplied (the
    Megatron-side scripts stamp one that identifies the exact build:
    ``sha256:megatron-core-r0.13-tp4-bf16-torch-dist``). Falls back to
    a derived fingerprint built from FSDP-shape defaults so the dense
    pipeline keeps working unchanged when the trainer payload omits
    the field.
    """
    engine_name = entry.get("engine", _DEFAULT_TRAINER_ENGINE_NAME)
    explicit = entry.get("engine_fingerprint")
    if isinstance(explicit, str) and explicit:
        fingerprint = explicit
    else:
        sharding = entry.get("sharding", "FULL_SHARD")
        dtype = entry.get("dtype", "bfloat16")
        attn = entry.get("attn_impl", "sdpa")
        world = entry.get("world_size", "?")
        fingerprint = (
            f"sha256:{engine_name}-{sharding.lower()}-{dtype}-{attn}-ws{world}"
        )
    return EngineFingerprint(
        name=engine_name,
        version=_DEFAULT_TRAINER_ENGINE_VERSION,
        fingerprint=fingerprint,
    )


def _tokenizer_hash_for(model_id: str) -> str:
    """Deterministic placeholder tokenizer-hash derived from ``model_id``.

    The real tokenizer hash is computed elsewhere (in the live
    pipeline); for offline pairing we just need a stable string so
    reports for the same model match. This is the same approach the
    existing single-prompt dense pipeline takes when the tokenizer hash
    isn't otherwise available on the rollout side.
    """
    return "sha256:" + sha256(model_id.encode("utf-8")).hexdigest()[:16]


def pair_reports(
    *,
    index: dict,
    trainers: list[dict],
    trajectories_by_run_id: dict[str, AgentTrajectory],
    precision_class: str,
    trainer_engine_label: str | None = None,
) -> list[DenseMismatchReport]:
    """Join an index + trainer outputs into one report per (trajectory, turn).

    Pure function — no I/O. Raises ``ValueError`` on:
      * missing ``prompt_idx`` in the trainer list,
      * length mismatch between the captured ``response_logprobs`` and
        the trainer's ``trainer_logprobs`` (would yield a malformed
        ``DenseMismatchReport`` from ``compute_dense_mismatch``).
    """
    entries: list[dict] = index["entries"]
    trainers_by_idx: dict[int, dict] = {t["prompt_idx"]: t for t in trainers}

    reports: list[DenseMismatchReport] = []
    for entry in entries:
        prompt_idx = entry["prompt_idx"]
        if prompt_idx not in trainers_by_idx:
            raise ValueError(
                f"trainer entry missing for prompt_idx={prompt_idx} "
                f"(task_id={entry['task_id']}, turn={entry['assistant_turn_idx']})"
            )
        t_entry = trainers_by_idx[prompt_idx]
        trainer_lp = t_entry["trainer_logprobs"]
        rollout_lp = entry["response_logprobs"]
        if len(trainer_lp) != len(rollout_lp):
            raise ValueError(
                f"length mismatch at prompt_idx={prompt_idx}: "
                f"rollout has {len(rollout_lp)} logprobs, "
                f"trainer has {len(trainer_lp)}"
            )

        traj = trajectories_by_run_id.get(entry["trajectory_run_id"])
        if traj is None:
            raise ValueError(
                f"trajectory_run_id={entry['trajectory_run_id']!r} not found "
                "among supplied trajectories"
            )
        rollout_engine = traj.rollout_engine
        trainer_engine = _trainer_engine_from_entry(t_entry)
        if trainer_engine_label is not None:
            trainer_engine = EngineFingerprint(
                name=trainer_engine_label,
                version=trainer_engine.version,
                fingerprint=trainer_engine.fingerprint,
            )

        prompt_id = f"{entry['task_id']}-turn{entry['assistant_turn_idx']}"
        run_id = f"{rollout_engine.name}-vs-{trainer_engine.name}-{prompt_id}"
        report = compute_dense_mismatch(
            run_id=run_id,
            model_id=traj.model_id,
            checkpoint_digest=_tokenizer_hash_for(traj.model_id),
            tokenizer_hash=_tokenizer_hash_for(traj.model_id),
            rollout_engine=rollout_engine,
            trainer_engine=trainer_engine,
            precision_class=precision_class,
            prompt_id=prompt_id,
            rollout_logprobs=list(rollout_lp),
            trainer_logprobs=list(trainer_lp),
            notes=[
                f"hermes-agent dense pair: trajectory_run_id="
                f"{entry['trajectory_run_id']!r}, "
                f"assistant_turn_idx={entry['assistant_turn_idx']}"
            ],
        )
        reports.append(report)
    return reports


def write_reports(
    reports: list[DenseMismatchReport],
    out_root: Path,
) -> list[Path]:
    """Write one ``dense_mismatch_report.json`` per report under ``out_root``.

    The per-report subdirectory is named ``<prompt_id>`` (which is
    ``<task_id>-turn<k>``); the dashboard renderer globs across these.
    """
    written: list[Path] = []
    out_root.mkdir(parents=True, exist_ok=True)
    for r in reports:
        sub = out_root / r.prompt_id
        sub.mkdir(parents=True, exist_ok=True)
        path = sub / "dense_mismatch_report.json"
        path.write_text(r.model_dump_json(indent=2), encoding="utf-8")
        written.append(path)
    return written


def _load_trajectories(traj_dir: Path) -> dict[str, AgentTrajectory]:
    files = sorted(traj_dir.glob("*.json"))
    out: dict[str, AgentTrajectory] = {}
    for p in files:
        traj = load_trajectory(p)
        out[traj.run_id] = traj
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pair build_hermes_dense_input index + run_fsdp_reference outputs "
            "into one DenseMismatchReport per (trajectory, assistant_turn)."
        )
    )
    parser.add_argument("--index", required=True, help="Path to the index JSON.")
    parser.add_argument(
        "--trainers", required=True, help="Path to the trainer-side /tmp/trainers.json."
    )
    parser.add_argument(
        "--trajectory-dir",
        required=True,
        help="Directory of the AgentTrajectory JSONs whose probes seeded the index.",
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Directory under which per-report subdirs are written.",
    )
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
    if not index_path.exists():
        print(f"[fatal] index not found: {index_path}", file=sys.stderr)
        return 2
    if not trainers_path.exists():
        print(f"[fatal] trainers not found: {trainers_path}", file=sys.stderr)
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
        f"[hermes-pair] {len(reports)} report(s) written under {out_root}",
        flush=True,
    )
    for p in written:
        print(f"  - {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
