from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import (
    BudgetReport,
    DecisionReason,
    GroupDecision,
    GroupStatus,
    SampleGroup,
)


@dataclass(frozen=True)
class BudgetPolicy:
    """Thresholds for the off-policy budget controller.

    The graduated path is:
      veto -> quarantine
      hard ESS / lag fail -> quarantine
      high clipped_fraction -> train_with_correction
      moderate ESS or lag -> replay
      otherwise -> train
    """

    clamp: float = 20.0
    min_ess: float = 0.30
    replay_ess_threshold: float = 0.60
    max_clipped_fraction: float = 0.10
    max_policy_lag_steps: int = 8
    replay_lag_threshold: int = 4
    veto_abs_log_ratio: float = 30.0
    # MoE-only quarantine gate. The Dense + MoE dashboards both flag
    # router_flip_rate > 15% as the marketplace red threshold, so we
    # mirror that here. ``None`` on a BudgetReport (Dense rollout)
    # skips this check entirely.
    max_router_flip_rate: float = 0.15


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
    *,
    router_flip_rate: float | None = None,
) -> BudgetReport:
    """Compute budget metrics for one group.

    `train_logprobs` should correspond to the same valid policy tokens selected by
    action_mask == 1 and tool_output_mask == 0.

    ``router_flip_rate`` is the optional MoE router-mismatch signal
    (rollout-vs-trainer top-1 expert disagreement fraction over the
    same response tokens). When supplied it threads onto the
    ``BudgetReport`` and ``decide_group`` quarantines the group if it
    exceeds ``BudgetPolicy.max_router_flip_rate``. Dense rollouts
    leave this ``None`` and the gate is skipped.
    """
    policy = policy or BudgetPolicy()
    rollout = _flatten_rollout_logprobs(group)
    train = np.asarray(train_logprobs, dtype=np.float64)
    if rollout.shape != train.shape:
        raise ValueError(
            f"train_logprobs shape {train.shape} must match rollout tokens {rollout.shape}"
        )
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
    if router_flip_rate is not None and router_flip_rate > policy.max_router_flip_rate:
        notes.append("router_flip_rate exceeds limit")

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
        router_flip_rate=router_flip_rate,
        notes=notes,
    )


def _format_reasons(reasons: list[DecisionReason]) -> str:
    return ", ".join(r.value for r in reasons)


def decide_group(report: BudgetReport, policy: BudgetPolicy | None = None) -> GroupDecision:
    """Map a BudgetReport to a typed decision.

    Severity ordering: veto > quarantine (hard ESS / lag) > correctable (high
    clipped) > replay (moderate ESS or lag) > train. Each branch attaches the
    typed DecisionReason(s) that motivated it; consumers can read the enum
    values without parsing free-text.
    """
    policy = policy or BudgetPolicy()

    if report.veto_fraction > 0:
        reasons = [DecisionReason.VETO_THRESHOLD_EXCEEDED]
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.QUARANTINED,
            reason=_format_reasons(reasons),
            decision_reasons=reasons,
            budget_report=report,
            recommended_action="quarantine",
        )

    quarantine_reasons: list[DecisionReason] = []
    if report.policy_lag_steps > policy.max_policy_lag_steps:
        quarantine_reasons.append(DecisionReason.STALE_POLICY_LAG)
    if report.effective_sample_size < policy.min_ess:
        quarantine_reasons.append(DecisionReason.LOW_ESS)
    if (
        report.router_flip_rate is not None
        and report.router_flip_rate > policy.max_router_flip_rate
    ):
        # MoE-only: top-1 expert disagreement above threshold means
        # too many tokens route through the wrong expert sub-network
        # — corrupts the gradient on those tokens regardless of how
        # well logprob-level off-policy correction works.
        quarantine_reasons.append(DecisionReason.HIGH_ROUTER_FLIP_RATE)
    if quarantine_reasons:
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.QUARANTINED,
            reason=_format_reasons(quarantine_reasons),
            decision_reasons=quarantine_reasons,
            budget_report=report,
            recommended_action="quarantine",
        )

    if report.clipped_fraction > policy.max_clipped_fraction:
        reasons = [DecisionReason.HIGH_CLIPPED_FRACTION]
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.CORRECTABLE,
            reason=_format_reasons(reasons),
            decision_reasons=reasons,
            budget_report=report,
            recommended_action="train_with_correction",
        )

    replay_reasons: list[DecisionReason] = []
    if report.policy_lag_steps > policy.replay_lag_threshold:
        replay_reasons.append(DecisionReason.REPLAY_TIER_LAG)
    if report.effective_sample_size < policy.replay_ess_threshold:
        replay_reasons.append(DecisionReason.REPLAY_TIER_ESS)
    if replay_reasons:
        return GroupDecision(
            group_id=report.group_id,
            status=GroupStatus.ACCEPTED,
            reason=_format_reasons(replay_reasons),
            decision_reasons=replay_reasons,
            budget_report=report,
            recommended_action="replay",
        )

    reasons = [DecisionReason.WITHIN_BUDGET]
    return GroupDecision(
        group_id=report.group_id,
        status=GroupStatus.ACCEPTED,
        reason=_format_reasons(reasons),
        decision_reasons=reasons,
        budget_report=report,
        recommended_action="train",
    )
