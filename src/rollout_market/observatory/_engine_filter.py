"""Render-time engine filter for the public dashboards.

Canonical architecture: **vLLM as the only inference (rollout) engine,
FSDP and Megatron as the "mock trainers" (forward-only references that
stand in for a real trainer's forward pass).** Each model (dense
Qwen3-32B / MoE Qwen3-30B-A3B) shows up at multiple rollout precisions
(bf16, fp8) but the trainer side stays bf16 (the normal training
regime — quantization belongs to inference, not training).

* **Headline** — only `vllm` rollouts paired with `fsdp` or `megatron`
  trainer-refs. Everything else is hidden from the public dashboards.
* **Hidden** — sglang rollouts and HF-as-engine rows. Per the
  ``cleanup.purge_inference_only`` redirect we no longer render them
  on the public surface; ingestion still loads them (so historical
  JSON survives), they just don't reach the rendered HTML.

The filter is render-time only: ingestion (``build_dashboard`` /
``as_dict``) keeps every report so JSON consumers still see the full
dataset.
"""

from __future__ import annotations

APPENDIX_HEADER = "All engines (full data)"

_HEADLINE_ROLLOUTS = ("vllm",)
_HEADLINE_TRAINERS = ("fsdp", "megatron")
_BLOCKED_ROLLOUTS = ("sglang",)
_BLOCKED_TRAINERS = ("hf-transformers", "hf")


def _norm(name: str) -> str:
    return (name or "").lower()


def is_headline_rollout(rollout_engine: str) -> bool:
    """True if the rollout engine name should appear on the headline."""
    n = _norm(rollout_engine)
    return any(tag in n for tag in _HEADLINE_ROLLOUTS)


def is_headline_trainer(trainer_engine: str) -> bool:
    """True if the trainer-side reference should appear on the headline."""
    n = _norm(trainer_engine)
    return any(tag in n for tag in _HEADLINE_TRAINERS)


def is_headline_pair(rollout_engine: str, trainer_engine: str) -> bool:
    """True if a (rollout, trainer) pair belongs on the headline."""
    return is_headline_rollout(rollout_engine) and is_headline_trainer(trainer_engine)


def is_blocked_pair(rollout_engine: str, trainer_engine: str) -> bool:
    """True if a (rollout, trainer) pair is purged from the public surface.

    sglang rollouts and HF-as-trainer rows are kept in the underlying
    JSON for historical reasons but never reach the rendered HTML on
    /docs/. Use this to drop appendix rendering entirely.
    """
    r = _norm(rollout_engine)
    t = _norm(trainer_engine)
    if any(tag in r for tag in _BLOCKED_ROLLOUTS):
        return True
    if any(tag in t for tag in _BLOCKED_TRAINERS):
        return True
    return False
