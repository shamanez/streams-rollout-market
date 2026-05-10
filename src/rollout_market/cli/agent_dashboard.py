"""CLI: aggregate agent_divergence_report.json files into one dashboard.

Usage::

    python -m rollout_market.cli.agent_dashboard \\
        --reports-glob 'runs/live/agent_diff/*/agent_divergence_report.json' \\
        --out-dir runs/live/agent_dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..observatory.agent_dashboard import (
    build_dashboard,
    load_reports_glob,
    write_dashboard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-dashboard",
        description="Aggregate agent_divergence_report.json files into one HTML+JSON dashboard.",
    )
    parser.add_argument(
        "--reports-glob",
        required=True,
        help="Glob matching agent_divergence_report.json files",
    )
    parser.add_argument("--out-dir", required=True, help="Where to write the dashboard files")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reports = load_reports_glob(args.reports_glob)
    if not reports:
        print(f"warning: no reports matched {args.reports_glob!r}", file=sys.stderr)
    dashboard = build_dashboard(reports)
    paths = write_dashboard(dashboard, Path(args.out_dir))
    print(str(paths["json"]))
    print(str(paths["html"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
