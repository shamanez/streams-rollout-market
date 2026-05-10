"""CLI: compute one trajectory-divergence report from two AgentTrajectory files.

Usage::

    python -m rollout_market.cli.agent_trajectory_lab \\
        --rollout runs/live/agent/vllm-fp8/task-research-001/trajectory.json \\
        --trainer runs/live/agent/vllm-bf16/task-research-001/trajectory.json \\
        --out-root runs/live/agent_diff
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..observatory.agent_trajectory_lab import (
    compute_trajectory_divergence,
    load_trajectory,
    write_trajectory_diff,
)
from ._runs import run_dir_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-trajectory-lab",
        description=(
            "Compute trajectory-divergence (first-divergence step, tool choice "
            "disagreement, jaccard, answer match) between two AgentTrajectory "
            "JSON files on the same task."
        ),
    )
    parser.add_argument("--rollout", required=True, help="rollout-side AgentTrajectory JSON")
    parser.add_argument("--trainer", required=True, help="trainer-side AgentTrajectory JSON")
    parser.add_argument("--out-root", required=True, help="root dir for the output report")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rollout = load_trajectory(args.rollout)
    trainer = load_trajectory(args.trainer)
    report = compute_trajectory_divergence(rollout, trainer)
    out_dir = Path(args.out_root) / run_dir_name()
    paths = write_trajectory_diff(report, rollout, trainer, out_dir)
    print(str(paths["json"]))
    print(str(paths["html"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
