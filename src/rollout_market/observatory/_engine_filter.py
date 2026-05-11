"""Render-time engine filter for the public dashboards.

STEER directive ``dashboard.engine_filter`` carves the public dashboard
surface into two zones:

* **Headline** — only vLLM as the rollout engine, and FSDP or Megatron
  as the trainer-side reference. This is the comparison a CTO-level
  reader is supposed to see first.
* **Appendix** — everything else (sglang as rollout, HF as either side,
  any unknown engine). The appendix is collapsible by default and lives
  underneath the headline.

The filter is render-time only: ingestion (``build_dashboard`` /
``as_dict``) keeps every report so JSON consumers still see the full
dataset.
"""

from __future__ import annotations

APPENDIX_HEADER = "All engines (full data)"

_HEADLINE_ROLLOUTS = ("vllm",)
_HEADLINE_TRAINERS = ("fsdp", "megatron")


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
