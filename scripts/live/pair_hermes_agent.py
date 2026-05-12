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


def _list_trajectories(dirs: list[Path]) -> dict[str, list[Path]]:
    """Return ``{task_id: [trajectory_json, ...]}`` across all replicate dirs.

    A directory may contain multiple replicates of the same task — the
    files just need to be parseable AgentTrajectory JSON and share the
    same ``task_id`` field. We bucket by **filename stem** because
    hermes_agent_runner writes one trajectory per task with the file
    name = task_id; replicate dirs follow the same convention. Empty
    or missing directories are skipped silently.
    """
    out: dict[str, list[Path]] = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.json")):
            out.setdefault(path.stem, []).append(path)
    return out


def _pair_side(
    *,
    rollout_dirs: list[Path],
    trainer_dirs: list[Path],
    out_root: Path,
) -> int:
    """Pair every rollout replicate against every trainer replicate per task.

    n_rollout_replicates x n_trainer_replicates pairs are produced per
    task. With one replicate per side this collapses to the original
    1-pair-per-task behaviour. The aggregator can then average over a
    much larger sample for noise-floor / precision-effect tiles.
    """
    rollout_by_task = _list_trajectories(rollout_dirs)
    trainer_by_task = _list_trajectories(trainer_dirs)
    if not rollout_by_task or not trainer_by_task:
        print(
            f"[warn] skipping pair: empty rollout {[str(d) for d in rollout_dirs]} "
            f"or trainer {[str(d) for d in trainer_dirs]}",
            file=sys.stderr,
        )
        return 0
    paired = 0
    for task_id in sorted(rollout_by_task.keys() & trainer_by_task.keys()):
        for r in rollout_by_task[task_id]:
            for t in trainer_by_task[task_id]:
                if r == t:
                    continue  # noise-floor against itself (replicate vs itself)
                _pair_one(r, t, out_root)
                paired += 1
    return paired


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pair_hermes_agent")
    parser.add_argument("--model-slug", required=True, help="e.g. qwen3-32b or qwen3-30b-a3b")
    parser.add_argument(
        "--bf16-dir",
        required=True,
        action="append",
        help=(
            "bf16 trajectory dir; repeat the flag to pass multiple replicate "
            "dirs (e.g. --bf16-dir runs/.../bf16 --bf16-dir runs/.../bf16_rep2)"
        ),
    )
    parser.add_argument(
        "--fp8-dir",
        required=False,
        action="append",
        default=None,
        help="fp8 trajectory dir; repeatable for replicates",
    )
    parser.add_argument(
        "--bf16-seedb-dir",
        required=False,
        action="append",
        default=None,
        help="bf16_seedB trajectory dir (noise floor); repeatable for replicates",
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

    bf16_dirs = [Path(d) for d in (args.bf16_dir or [])]
    out_base = Path(args.out_base)

    total_paired = 0
    if args.fp8_dir:
        fp8_dirs = [Path(d) for d in args.fp8_dir]
        out = out_base / f"hermes-{args.model_slug}-fp8-vs-bf16"
        out.mkdir(parents=True, exist_ok=True)
        n = _pair_side(rollout_dirs=fp8_dirs, trainer_dirs=bf16_dirs, out_root=out)
        print(f"[paired] fp8-vs-bf16: {n} pairs (across replicates)")
        total_paired += n

    if args.bf16_seedb_dir:
        sb_dirs = [Path(d) for d in args.bf16_seedb_dir]
        out = out_base / f"hermes-{args.model_slug}-bf16_seedB-vs-bf16"
        out.mkdir(parents=True, exist_ok=True)
        n = _pair_side(rollout_dirs=sb_dirs, trainer_dirs=bf16_dirs, out_root=out)
        print(f"[paired] bf16_seedB-vs-bf16: {n} pairs (across replicates)")
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
