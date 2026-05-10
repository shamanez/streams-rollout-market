"""Shared helper for naming run output directories.

Concurrent CLI invocations that finish in the same UTC second collide
on a plain ``<UTC-ts>`` directory name, with the second writer
overwriting the first. ``run_dir_name()`` appends a short uuid4 hex
suffix so the second resolution is preserved when matters and a fresh
4-hex tag separates parallel invocations otherwise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


def run_dir_name() -> str:
    """Return ``<YYYYMMDDTHHMMSSZ>-<4-hex>`` — a per-invocation unique label.

    The UUID suffix has 16^4 = 65,536 distinct values; under any
    realistic concurrent-CLI workload the collision probability per
    second is negligible. The timestamp is still leading so directory
    listings sort chronologically.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid4().hex[:4]
    return f"{ts}-{suffix}"
