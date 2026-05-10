"""CLI: compute and render one dense mismatch report from a fixture file.

The fixture is a JSON file mirroring the keyword arguments of
``compute_dense_mismatch`` (rollout_logprobs, trainer_logprobs, identity
fields, etc.). The CLI does not run any model — paired logprob streams are
the input.

Usage::

    python -m rollout_market.cli.dense_mismatch_lab \\
        --input examples/dense_mismatch_input.json \\
        --out-root runs

Writes ``runs/<UTC-timestamp>/dense_mismatch_report.{json,html}``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..observatory.dense_mismatch_lab import (
    EngineFingerprint,
    compute_dense_mismatch,
    load_input_fixture,
    write_dense_mismatch,
)
from ._runs import run_dir_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dense-mismatch-lab",
        description=(
            "Compute the mismatch-observatory metrics contract from a paired "
            "rollout/trainer logprob fixture. Trainer-free."
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fixture = load_input_fixture(args.input)
    fixture["rollout_engine"] = _coerce_engine(fixture["rollout_engine"])
    fixture["trainer_engine"] = _coerce_engine(fixture["trainer_engine"])

    report = compute_dense_mismatch(**fixture)

    out_dir = Path(args.out_root) / run_dir_name()
    paths = write_dense_mismatch(report, out_dir)
    print(str(paths["json"]))
    print(str(paths["html"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
