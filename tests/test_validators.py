import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    PolicyManifest,
    RejectionReason,
    RolloutLease,
    SampleGroup,
)
from rollout_market.validators import validate_group_against_lease

EXAMPLES = Path(__file__).parents[1] / "examples"


def _load_group() -> SampleGroup:
    return SampleGroup.model_validate(json.loads((EXAMPLES / "minimal_grpo_group.json").read_text()))


def _make_manifest(group: SampleGroup) -> PolicyManifest:
    return PolicyManifest(
        policy_version=group.policy_version,
        checkpoint_digest=group.checkpoint_digest,
        tokenizer_hash=group.tokenizer_hash,
        model_config_hash="cfg-test",
        precision_class=group.precision_class,
        quantization_class=group.quantization_class,
    )


def _make_lease(group: SampleGroup, **overrides) -> RolloutLease:
    issued = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
    base = dict(
        lease_id="lease-1",
        job_id="job-1",
        assignment_id=group.assignment_id,
        worker_id=group.worker_id,
        policy_version=group.policy_version,
        required_precision_class=group.precision_class,
        required_quantization_class=group.quantization_class,
        max_context_tokens=4096,
        group_size=len(group.trajectories),
        idempotency_key="idem-1",
        issued_at=issued,
        expires_at=issued + timedelta(minutes=10),
    )
    base.update(overrides)
    return RolloutLease.model_validate(base)


def _now() -> datetime:
    return datetime(2026, 5, 10, 0, 0, 30, tzinfo=timezone.utc)


def test_clean_group_returns_none():
    group = _load_group()
    lease = _make_lease(group)
    manifest = _make_manifest(group)
    assert validate_group_against_lease(group, lease, manifest, now=_now()) is None


def test_expired_lease_is_rejected():
    group = _load_group()
    issued = datetime(2026, 5, 10, tzinfo=timezone.utc)
    lease = _make_lease(group, issued_at=issued, expires_at=issued + timedelta(seconds=10))
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(
        group, lease, manifest, now=issued + timedelta(minutes=1)
    )
    assert decision is not None
    assert decision.status == GroupStatus.REJECTED
    assert decision.rejection_reason == RejectionReason.EXPIRED_LEASE
    assert decision.recommended_action == "reject"


def test_policy_version_mismatch_against_lease():
    group = _load_group()
    lease = _make_lease(group, policy_version="v9999")
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.POLICY_VERSION_MISMATCH


def test_mixed_policy_version_against_manifest():
    group = _load_group()
    lease = _make_lease(group)
    manifest = _make_manifest(group).model_copy(update={"policy_version": "v9999"})
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.MIXED_POLICY_VERSION


def test_checkpoint_mismatch():
    group = _load_group()
    lease = _make_lease(group)
    manifest = _make_manifest(group).model_copy(update={"checkpoint_digest": "sha256:other"})
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.CHECKPOINT_MISMATCH


def test_tokenizer_mismatch():
    group = _load_group()
    lease = _make_lease(group)
    manifest = _make_manifest(group).model_copy(update={"tokenizer_hash": "tok-other"})
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.TOKENIZER_MISMATCH


def test_precision_mismatch():
    group = _load_group()
    lease = _make_lease(group, required_precision_class="bf16")  # group is fp32
    manifest = _make_manifest(group).model_copy(update={"precision_class": "bf16"})
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.PRECISION_MISMATCH


def test_quantization_mismatch():
    group = _load_group()
    lease = _make_lease(group, required_quantization_class="awq-int4")
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.QUANTIZATION_MISMATCH


def test_group_size_mismatch():
    group = _load_group()
    lease = _make_lease(group, group_size=99)
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.GROUP_SIZE_MISMATCH


def test_missing_token_ids_in_prompt():
    group = _load_group().model_copy(update={"prompt_token_ids": []})
    lease = _make_lease(group)
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.MISSING_TOKEN_IDS


def test_missing_logprobs_all_zero():
    group = _load_group()
    new_trajs = [
        t.model_copy(update={"rollout_logprobs": [0.0] * len(t.response_token_ids)})
        for t in group.trajectories
    ]
    group = group.model_copy(update={"trajectories": new_trajs})
    lease = _make_lease(group)
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.MISSING_LOGPROBS


def test_context_overflow():
    group = _load_group()
    lease = _make_lease(group, max_context_tokens=2)
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(group, lease, manifest, now=_now())
    assert decision.rejection_reason == RejectionReason.CONTEXT_OVERFLOW


def test_group_decision_rejects_inconsistent_status():
    with pytest.raises(ValidationError):
        GroupDecision(
            group_id="g",
            status=GroupStatus.REJECTED,
            reason="missing reason",
            recommended_action="reject",
        )
    with pytest.raises(ValidationError):
        GroupDecision(
            group_id="g",
            status=GroupStatus.ACCEPTED,
            reason="ok",
            rejection_reason=RejectionReason.TOKENIZER_MISMATCH,
            recommended_action="train",
        )


def test_each_rejection_reason_has_distinct_value():
    values = [r.value for r in RejectionReason]
    assert len(values) == len(set(values))
    for required in (
        "mixed_policy_version",
        "missing_logprobs",
        "missing_token_ids",
        "tokenizer_mismatch",
        "precision_mismatch",
        "expired_lease",
    ):
        assert required in values
