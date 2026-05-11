"""Device fingerprint contract for the mismatch labs.

STEER ``dashboard.device_axis`` introduces a `DeviceFingerprint`
Pydantic model that captures the GPU / host the rollout (or trainer
reference) was generated on. The field is backwards-compatible —
existing reports were written without it, so every report model
defaults ``device`` to ``None``. Dashboards pivot by device and treat
``device=None`` as the legacy spot-instance bucket
``L40S (g6e.12xlarge)``.

The CLIs read the fingerprint from environment variables so the live
scripts (`scripts/live/*.py`) can stamp every report without needing a
new flag on every script::

    DEVICE_VENDOR=NVIDIA \
    DEVICE_FAMILY=L40S \
    DEVICE_CUDA_COMPUTE=8.9 \
    DEVICE_VRAM_GB=24 \
    DEVICE_HOST_LABEL='g6e.12xlarge' \
    python -m rollout_market.cli.dense_mismatch_lab ...

If none of the env vars are set, ``device_from_env`` returns ``None``
and the report keeps its legacy shape.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel


DEFAULT_DEVICE_BUCKET: Literal["L40S (g6e.12xlarge)"] = "L40S (g6e.12xlarge)"

_ENV_VARS = (
    "DEVICE_VENDOR",
    "DEVICE_FAMILY",
    "DEVICE_CUDA_COMPUTE",
    "DEVICE_VRAM_GB",
    "DEVICE_HOST_LABEL",
)


class DeviceFingerprint(BaseModel):
    """The GPU + host a report was generated on."""

    vendor: str
    family: str
    cuda_compute: str | None = None
    vram_gb: int | None = None
    host_label: str | None = None


def device_from_env(env: dict[str, str] | None = None) -> DeviceFingerprint | None:
    """Build a DeviceFingerprint from the standard env vars, or return None."""
    env = env if env is not None else os.environ
    if not any(env.get(name) for name in _ENV_VARS):
        return None
    vram_raw = env.get("DEVICE_VRAM_GB")
    vram = int(vram_raw) if vram_raw else None
    return DeviceFingerprint(
        vendor=env.get("DEVICE_VENDOR", "unknown"),
        family=env.get("DEVICE_FAMILY", "unknown"),
        cuda_compute=env.get("DEVICE_CUDA_COMPUTE") or None,
        vram_gb=vram,
        host_label=env.get("DEVICE_HOST_LABEL") or None,
    )


def device_bucket_label(device: DeviceFingerprint | None) -> str:
    """Dashboard-display label. Falls back to the legacy bucket on None."""
    if device is None:
        return DEFAULT_DEVICE_BUCKET
    parts = [device.family]
    if device.host_label:
        parts.append(f"({device.host_label})")
    return " ".join(parts)
