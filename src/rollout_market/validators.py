"""Pre-OPBC contract validation.

Maps a (SampleGroup, RolloutLease, PolicyManifest) triple to a typed
GroupDecision. Returns None when the group is contract-clean and should
proceed to OPBC budget scoring; otherwise returns a REJECTED decision with
a single concrete `rejection_reason`.

Checks here are about contract integrity (the rollout describes what the
job asked for) rather than budget quality (how off-policy the rollout is).
"""

from __future__ import annotations

from datetime import datetime, timezone

from .contracts import (
    GroupDecision,
    GroupStatus,
    PolicyManifest,
    RejectionReason,
    RolloutLease,
    SampleGroup,
)


def _reject(group_id: str, reason: RejectionReason, message: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.REJECTED,
        reason=message,
        rejection_reason=reason,
        recommended_action="reject",
    )


def validate_group_against_lease(
    group: SampleGroup,
    lease: RolloutLease,
    manifest: PolicyManifest,
    now: datetime | None = None,
) -> GroupDecision | None:
    """Return a typed REJECTED decision, or None if the group is contract-clean.

    Order is intentionally fixed so the *first* violated invariant determines
    the rejection reason — this keeps rejection telemetry attributable.
    """
    ref_now = now if now is not None else datetime.now(timezone.utc)

    if lease.is_expired(ref_now):
        return _reject(
            group.group_id,
            RejectionReason.EXPIRED_LEASE,
            f"lease {lease.lease_id} expired at {lease.expires_at.isoformat()}",
        )

    if group.policy_version != lease.policy_version:
        return _reject(
            group.group_id,
            RejectionReason.POLICY_VERSION_MISMATCH,
            f"group policy_version={group.policy_version!r} != lease {lease.policy_version!r}",
        )

    if group.policy_version != manifest.policy_version:
        return _reject(
            group.group_id,
            RejectionReason.MIXED_POLICY_VERSION,
            f"group policy_version={group.policy_version!r} != manifest {manifest.policy_version!r}",
        )

    if group.checkpoint_digest != manifest.checkpoint_digest:
        return _reject(
            group.group_id,
            RejectionReason.CHECKPOINT_MISMATCH,
            f"checkpoint_digest {group.checkpoint_digest!r} != manifest {manifest.checkpoint_digest!r}",
        )

    if group.tokenizer_hash != manifest.tokenizer_hash:
        return _reject(
            group.group_id,
            RejectionReason.TOKENIZER_MISMATCH,
            f"tokenizer_hash {group.tokenizer_hash!r} != manifest {manifest.tokenizer_hash!r}",
        )

    if group.precision_class != lease.required_precision_class:
        return _reject(
            group.group_id,
            RejectionReason.PRECISION_MISMATCH,
            f"precision_class {group.precision_class!r} != required {lease.required_precision_class!r}",
        )

    if (lease.required_quantization_class or None) != (group.quantization_class or None):
        return _reject(
            group.group_id,
            RejectionReason.QUANTIZATION_MISMATCH,
            f"quantization_class {group.quantization_class!r} != required "
            f"{lease.required_quantization_class!r}",
        )

    if len(group.trajectories) != lease.group_size:
        return _reject(
            group.group_id,
            RejectionReason.GROUP_SIZE_MISMATCH,
            f"group has {len(group.trajectories)} trajectories, lease requires {lease.group_size}",
        )

    if not group.prompt_token_ids:
        return _reject(
            group.group_id,
            RejectionReason.MISSING_TOKEN_IDS,
            "prompt_token_ids is empty",
        )

    for idx, traj in enumerate(group.trajectories):
        if not traj.response_token_ids:
            return _reject(
                group.group_id,
                RejectionReason.MISSING_TOKEN_IDS,
                f"trajectory {idx} has empty response_token_ids",
            )
        if not traj.rollout_logprobs or all(lp == 0.0 for lp in traj.rollout_logprobs):
            return _reject(
                group.group_id,
                RejectionReason.MISSING_LOGPROBS,
                f"trajectory {idx} is missing rollout_logprobs",
            )

    longest = max(
        (len(group.prompt_token_ids) + len(t.response_token_ids) for t in group.trajectories),
        default=0,
    )
    if longest > lease.max_context_tokens:
        return _reject(
            group.group_id,
            RejectionReason.CONTEXT_OVERFLOW,
            f"longest sequence {longest} exceeds lease max_context_tokens {lease.max_context_tokens}",
        )

    return None
