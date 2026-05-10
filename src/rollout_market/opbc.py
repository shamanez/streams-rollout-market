from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contracts import BudgetReport, GroupDecision, GroupStatus, SampleGroup


@dataclass(frozen=True)
class BudgetPolicy:
    clamp: float = 20.0
    min_ess: float = 0.30
    max_clipped_fraction: float = 0.10
    max_policy_lag_steps: int = 8
    veto_abs_log_ratio: float = 30.0


def _flatten_rollout_logprobs(group: SampleGroup) -> np.ndarray:
    values: list[float] = []
    for traj in group.trajectories:
        for lp, action, tool in zip(traj.rollout_logprobs, traj.action_mask, traj.tool_output_mask):
            if action and not tool:
                values.append(lp)
    return np.asarray(values, dtype=np.float64)


def compute_budget_report(
    group: SampleGroup,
    train_logprobs: list[float] | np.ndarray,
    policy: BudgetPolicy | None = None,
) -> BudgetReport:
    """Compute budget metrics for one group.

    `train_logprobs` should correspond to the same valid policy tokens selected by
    action_mask == 1 and tool_output_mask == 0.
    """
    policy = policy or BudgetPolicy()
    rollout = _flatten_rollout_logprobs(group)
    train = np.asarray(train_logprobs, dtype=np.float64)
    if rollout.shape != train.shape:
        raise ValueError(f"train_logprobs shape {train.shape} must match rollout tokens {rollout.shape}")
    if rollout.size == 0:
        raise ValueError("no valid policy tokens found after masks")

    log_ratio = train - rollout
    clipped = np.clip(log_ratio, -policy.clamp, policy.clamp)
    weights = np.exp(clipped)
    weight_sum = float(np.sum(weights))
    ess = float((weight_sum * weight_sum) / (len(weights) * np.sum(weights * weights)))
    clipped_fraction = float(np.mean(np.abs(log_ratio) > policy.clamp))
    veto_fraction = float(np.mean(np.abs(log_ratio) > policy.veto_abs_log_ratio))

    notes: list[str] = []
    if ess < policy.min_ess:
        notes.append("low effective sample size")
    if clipped_fraction > policy.max_clipped_fraction:
        notes.append("high clipped fraction")
    if group.policy_lag_steps > policy.max_policy_lag_steps:
        notes.append("policy lag exceeds limit")

    return BudgetReport(
        group_id=group.group_id,
        mean_weight=float(np.mean(weights)),
        std_weight=float(np.std(weights)),
        second_moment=float(np.mean(weights * weights)),
        effective_sample_size=ess,
        clipped_fraction=clipped_fraction,
        veto_fraction=veto_fraction,
        max_abs_log_ratio=float(np.max(np.abs(log_ratio))),
        policy_lag_steps=group.policy_lag_steps,
        notes=notes,
    )


def decide_group(report: BudgetReport, policy: BudgetPolicy | None = None) -> GroupDecision:
    policy = policy or BudgetPolicy()
    if report.veto_fraction > 0:
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.QUARANTINED,
            reason="veto threshold exceeded",
            budget_report=report,
            recommended_action="quarantine",
        )
    if report.policy_lag_steps > policy.max_policy_lag_steps or report.effective_sample_size < policy.min_ess:
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.QUARANTINED,
            reason="budget exceeds freshness or ESS limits",
            budget_report=report,
            recommended_action="quarantine",
        )
    if report.clipped_fraction > policy.max_clipped_fraction:
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.CORRECTABLE,
            reason="train only with correction and logging",
            budget_report=report,
            recommended_action="train_with_correction",
        )
    return GroupDecision(
        group_id=report.group_id,
        status=GroupStatus.ACCEPTED,
        reason="within budget",
        budget_report=report,
        recommended_action="train",
    )
