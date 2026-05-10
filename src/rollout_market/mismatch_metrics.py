from __future__ import annotations

from dataclasses import dataclass
from math import exp
from statistics import fmean


@dataclass(frozen=True)
class MismatchSummary:
    num_tokens: int
    delta_logprob_mean: float
    delta_logprob_abs_mean: float
    sequence_log_ratio: float
    ess: float
    second_moment: float
    clipped_fraction: float
    veto_fraction: float
    max_abs_log_ratio: float
    top_1pct_gradient_mass: float


def summarize_logprob_mismatch(
    rollout_logprobs: list[float],
    trainer_logprobs: list[float],
    mask: list[int] | None = None,
    *,
    clamp: float = 20.0,
    veto_abs_log_ratio: float = 30.0,
) -> MismatchSummary:
    """Summarize policy-gradient materiality for one sequence or group.

    rollout_logprobs are behavior-policy log probabilities. trainer_logprobs are
    recomputed under the trainer/current policy for the same token IDs. mask=1 means
    the token participates in policy-gradient accounting.
    """
    if len(rollout_logprobs) != len(trainer_logprobs):
        raise ValueError("rollout_logprobs and trainer_logprobs must have the same length")
    if mask is None:
        mask = [1] * len(rollout_logprobs)
    if len(mask) != len(rollout_logprobs):
        raise ValueError("mask length must match logprob lengths")

    deltas = [t - r for r, t, m in zip(rollout_logprobs, trainer_logprobs, mask) if m]
    if not deltas:
        raise ValueError("at least one token must be selected by mask")

    clipped = [max(-clamp, min(clamp, d)) for d in deltas]
    weights = [exp(d) for d in clipped]
    total = sum(weights)
    second_raw = sum(w * w for w in weights)
    ess = (total * total) / (len(weights) * second_raw)
    sorted_weights = sorted(weights, reverse=True)
    top_k = max(1, int(len(sorted_weights) * 0.01))

    return MismatchSummary(
        num_tokens=len(deltas),
        delta_logprob_mean=fmean(deltas),
        delta_logprob_abs_mean=fmean(abs(d) for d in deltas),
        sequence_log_ratio=sum(deltas),
        ess=ess,
        second_moment=second_raw / len(weights),
        clipped_fraction=sum(abs(d) > clamp for d in deltas) / len(deltas),
        veto_fraction=sum(abs(d) > veto_abs_log_ratio for d in deltas) / len(deltas),
        max_abs_log_ratio=max(abs(d) for d in deltas),
        top_1pct_gradient_mass=sum(sorted_weights[:top_k]) / total,
    )
