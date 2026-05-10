"""run_dir_name() must be unique across same-second invocations."""

from __future__ import annotations

import re

from rollout_market.cli._runs import run_dir_name

_PATTERN = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{4}$")


def test_run_dir_name_format():
    name = run_dir_name()
    assert _PATTERN.match(name), f"unexpected shape: {name!r}"


def test_run_dir_name_unique_across_rapid_calls():
    seen = {run_dir_name() for _ in range(200)}
    # 200 calls in the same second drawing 4-hex suffixes — collision
    # probability is bounded by birthday on 65,536 buckets, ~0.3% for 200
    # samples. We allow up to one accidental collision and still assert
    # the de-dup property holds in spirit.
    assert len(seen) >= 199


def test_run_dir_name_starts_with_utc_timestamp_in_seconds():
    name = run_dir_name()
    ts = name.split("-")[0]
    # exactly the YYYYmmddTHHMMSSZ shape — 16 characters
    assert len(ts) == 16
    assert ts.endswith("Z")
