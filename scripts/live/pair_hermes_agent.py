"""Pair Hermes Agent trajectories per-task and aggregate into one dashboard.

Walks ``runs/live/agent/hermes/<model_slug>/<side>/`` for the model/side
trajectory dirs produced by ``hermes_agent_runner.py``, runs
``compute_trajectory_divergence`` for each pair (precision-effect:
fp8-vs-bf16; noise-floor: bf16_seedB-vs-bf16), writes one
``agent_divergence_report.json`` per (pair, task) under
``runs/live/agent_diff/hermes-<model_slug>-<pair_slug>/<task>/`` and
finally re-renders the aggregate dashboard with
``rollout_market.cli.agent_dashboard``.

Usage::

    python scripts/live/pair_hermes_agent.py \\
        --model-slug qwen3-32b \\
        --bf16-dir runs/live/agent/hermes/qwen3-32b/bf16 \\
        --fp8-dir runs/live/agent/hermes/qwen3-32b/fp8 \\
        --bf16-seedb-dir runs/live/agent/hermes/qwen3-32b/bf16_seedB

The aggregate dashboard is written to
``runs/live/agent_dashboard/agent_dashboard.{html,json}`` so
``publish_dashboards.py`` can read it as the ``agent`` card payload.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _pair_one(rollout: Path, trainer: Path, out_root: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "rollout_market.cli.agent_trajectory_lab",
        "--rollout",
        str(rollout),
        "--trainer",
        str(trainer),
        "--out-root",
        str(out_root),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _pair_side(
    *,
    rollout_dir: Path,
    trainer_dir: Path,
    out_root: Path,
) -> int:
    if not rollout_dir.is_dir() or not trainer_dir.is_dir():
        print(
            f"[warn] skipping pair: missing {rollout_dir} or {trainer_dir}",
            file=sys.stderr,
        )
        return 0
    paired = 0
    for rollout_json in sorted(rollout_dir.glob("*.json")):
        task_id = rollout_json.stem
        trainer_json = trainer_dir / f"{task_id}.json"
        if not trainer_json.exists():
            print(f"[skip] {task_id}: no matching trainer trajectory", file=sys.stderr)
            continue
        _pair_one(rollout_json, trainer_json, out_root)
        paired += 1
    return paired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_hermes_agent")
    parser.add_argument("--model-slug", required=True, help="e.g. qwen3-32b or qwen3-30b-a3b")
    parser.add_argument("--bf16-dir", required=True, help="bf16 trajectory dir")
    parser.add_argument("--fp8-dir", required=False, default=None, help="fp8 trajectory dir")
    parser.add_argument(
        "--bf16-seedb-dir",
        required=False,
        default=None,
        help="bf16_seedB trajectory dir (noise floor)",
    )
    parser.add_argument(
        "--out-base",
        default="runs/live/agent_diff",
        help="where pair reports land",
    )
    parser.add_argument(
        "--dashboard-out",
        default="runs/live/agent_dashboard",
        help="where the aggregated dashboard lands",
    )
    args = parser.parse_args(argv)

    bf16 = Path(args.bf16_dir)
    out_base = Path(args.out_base)

    total_paired = 0
    if args.fp8_dir:
        fp8 = Path(args.fp8_dir)
        out = out_base / f"hermes-{args.model_slug}-fp8-vs-bf16"
        out.mkdir(parents=True, exist_ok=True)
        n = _pair_side(rollout_dir=fp8, trainer_dir=bf16, out_root=out)
        print(f"[paired] fp8-vs-bf16: {n} tasks")
        total_paired += n

    if args.bf16_seedb_dir:
        sb = Path(args.bf16_seedb_dir)
        out = out_base / f"hermes-{args.model_slug}-bf16_seedB-vs-bf16"
        out.mkdir(parents=True, exist_ok=True)
        n = _pair_side(rollout_dir=sb, trainer_dir=bf16, out_root=out)
        print(f"[paired] bf16_seedB-vs-bf16: {n} tasks")
        total_paired += n

    if total_paired == 0:
        print("[fatal] no pairs produced; refusing to re-render dashboard", file=sys.stderr)
        return 1

    print("[aggregating] running cli.agent_dashboard over all pairs in {0}".format(out_base))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rollout_market.cli.agent_dashboard",
            "--reports-glob",
            f"{out_base}/**/agent_divergence_report.json",
            "--out-dir",
            args.dashboard_out,
        ],
        cwd=str(REPO_ROOT),
        check=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
