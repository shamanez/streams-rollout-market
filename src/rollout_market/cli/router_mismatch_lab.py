"""CLI: compute and render one router-mismatch report from a fixture file.

The fixture is JSON mirroring ``compute_router_mismatch`` kwargs, with
``rollout_trace`` and ``trainer_trace`` as nested ``RouterTrace`` payloads.

Usage::

    python -m rollout_market.cli.router_mismatch_lab \\
        --input examples/router_mismatch_input.json \\
        --out-root runs

Writes ``runs/<UTC-timestamp>/router_mismatch_report.{json,html}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..observatory._device import device_from_env
from ..observatory.dense_mismatch_lab import EngineFingerprint
from ..observatory.router_mismatch_lab import (
    RouterTrace,
    compute_router_mismatch,
    load_input_fixture,
    write_router_mismatch,
)
from ._runs import run_dir_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="router-mismatch-lab",
        description=(
            "Compute router_flip_rate and token_expert_disagreement_rate from "
            "two aligned router traces. Trainer-free; no model run."
        ),
    )
    parser.add_argument("--input", required=True, help="Path to JSON fixture")
    parser.add_argument("--out-root", default="runs", help="Output root directory")
    return parser


def _coerce_engine(value: object) -> EngineFingerprint:
    if isinstance(value, EngineFingerprint):
        return value
    if isinstance(value, dict):
        return EngineFingerprint.model_validate(value)
    raise TypeError(f"engine fixture entry must be a dict, got {type(value).__name__}")


def _coerce_trace(value: object) -> RouterTrace:
    if isinstance(value, RouterTrace):
        return value
    if isinstance(value, dict):
        return RouterTrace.model_validate(value)
    raise TypeError(f"trace fixture entry must be a dict, got {type(value).__name__}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fixture = load_input_fixture(args.input)
    fixture["rollout_engine"] = _coerce_engine(fixture["rollout_engine"])
    fixture["trainer_engine"] = _coerce_engine(fixture["trainer_engine"])
    fixture["rollout_trace"] = _coerce_trace(fixture["rollout_trace"])
    fixture["trainer_trace"] = _coerce_trace(fixture["trainer_trace"])
    if "device" not in fixture:
        fixture["device"] = device_from_env()

    report = compute_router_mismatch(**fixture)

    out_dir = Path(args.out_root) / run_dir_name()
    paths = write_router_mismatch(report, out_dir)
    print(str(paths["json"]))
    print(str(paths["html"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
