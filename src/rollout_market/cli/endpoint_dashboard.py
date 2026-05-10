"""CLI: aggregate endpoint_contract_report.json files into one dashboard.

Usage::

    python -m rollout_market.cli.endpoint_dashboard \\
        --reports-glob 'runs/*/endpoint_contract_report.json' \\
        --out-dir runs/dashboards

Writes ``<out-dir>/endpoint_dashboard.{json,html}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..observatory.endpoint_dashboard import (
    build_dashboard,
    load_reports_glob,
    write_dashboard,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="endpoint-dashboard",
        description="Aggregate endpoint_contract_report.json files into one HTML+JSON dashboard.",
    )
    parser.add_argument(
        "--reports-glob",
        required=True,
        help="Glob pattern matching probe report JSON files (e.g. 'runs/*/endpoint_contract_report.json')",
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
