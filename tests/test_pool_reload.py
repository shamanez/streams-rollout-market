"""Phase 5.2: k-of-n reload quorum.

The tracker is a pure data structure; the more important property —
'per-group policy pinning remains strict even if pool-level sync is
quorum-based' — is exercised by feeding mismatched groups through
validators.validate_group_against_lease and asserting that the
tracker's existence does not loosen any rejection.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rollout_market.contracts import (
    PolicyManifest,
    RejectionReason,
    RolloutLease,
    SampleGroup,
)
from rollout_market.pool_reload import (
    PoolReloadTracker,
    QuorumViolation,
    ReloadAck,
)
from rollout_market.validators import validate_group_against_lease

EXAMPLES = Path(__file__).parents[1] / "examples"


def _ack(worker_id: str, version: str, digest: str = "sha256:demo") -> ReloadAck:
    return ReloadAck(worker_id=worker_id, policy_version=version, checkpoint_digest=digest)


def test_record_and_count():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("b", "v1"))
    tracker.record_ack(_ack("c", "v1"))
    assert tracker.ack_count("v1") == 3
    assert tracker.ack_count("v2") == 0


def test_workers_acked_returns_set():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("b", "v1"))
    assert tracker.workers_acked("v1") == {"a", "b"}


def test_re_ack_moves_worker_to_new_version():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("a", "v2"))
    assert tracker.ack_count("v1") == 0
    assert tracker.ack_count("v2") == 1


def test_has_quorum_uses_k():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("b", "v1"))
    assert tracker.has_quorum("v1", k=2) is True
    assert tracker.has_quorum("v1", k=3) is False


def test_has_quorum_rejects_invalid_k():
    tracker = PoolReloadTracker()
    with pytest.raises(ValueError):
        tracker.has_quorum("v1", k=0)


def test_has_quorum_rejects_n_smaller_than_k():
    tracker = PoolReloadTracker()
    with pytest.raises(ValueError):
        tracker.has_quorum("v1", k=3, n=2)


def test_pool_active_version_returns_none_when_no_quorum():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    assert tracker.pool_active_version(k=2) is None


def test_pool_active_version_picks_latest_with_quorum():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("b", "v1"))
    tracker.record_ack(_ack("c", "v2"))
    tracker.record_ack(_ack("d", "v2"))
    assert tracker.pool_active_version(k=2, candidates=["v2", "v1"]) == "v2"


def test_pool_active_version_falls_back_to_older_quorum():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("b", "v1"))
    tracker.record_ack(_ack("c", "v2"))  # only one v2 ack
    assert tracker.pool_active_version(k=2, candidates=["v2", "v1"]) == "v1"


def test_quorum_violation_on_disagreeing_digests():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1", digest="sha256:one"))
    with pytest.raises(QuorumViolation):
        tracker.record_ack(_ack("b", "v1", digest="sha256:two"))


def test_is_worker_aligned_gates_dispatch():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    assert tracker.is_worker_aligned("a", "v1") is True
    assert tracker.is_worker_aligned("a", "v2") is False
    assert tracker.is_worker_aligned("b", "v1") is False


# --- the load-bearing acceptance test --------------------------------------------


def _load_group() -> SampleGroup:
    return SampleGroup.model_validate(
        json.loads((EXAMPLES / "minimal_grpo_group.json").read_text())
    )


def _make_manifest(group: SampleGroup, *, version_override: str | None = None) -> PolicyManifest:
    return PolicyManifest(
        policy_version=version_override or group.policy_version,
        checkpoint_digest=group.checkpoint_digest,
        tokenizer_hash=group.tokenizer_hash,
        model_config_hash="cfg-test",
        precision_class=group.precision_class,
        quantization_class=group.quantization_class,
    )


def _make_lease(group: SampleGroup) -> RolloutLease:
    issued = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
    return RolloutLease(
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


def test_per_group_pinning_unchanged_when_pool_has_advanced():
    """Even if the pool has reached quorum on a *new* version, a stale
    group still gets rejected by the validator chain. Quorum advancement
    of the pool does not loosen per-group pinning at all."""
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v9999"))
    tracker.record_ack(_ack("b", "v9999"))
    assert tracker.has_quorum("v9999", k=2) is True

    group = _load_group()  # this fixture group is still on v0001
    lease = _make_lease(group)
    manifest = _make_manifest(group, version_override="v9999")  # registry is on v9999

    decision = validate_group_against_lease(
        group, lease, manifest, now=datetime(2026, 5, 10, 0, 0, 30, tzinfo=timezone.utc)
    )
    assert decision is not None
    assert decision.recommended_action == "reject"
    assert decision.rejection_reason == RejectionReason.MIXED_POLICY_VERSION


def test_unaligned_worker_with_stale_lease_still_rejected():
    """A worker that has acked v_new cannot use that ack to ship a v_old
    group whose lease is on v_old — validators still trip on the manifest
    mismatch."""
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v_new"))
    assert tracker.is_worker_aligned("a", "v_new") is True
    assert tracker.is_worker_aligned("a", "v_old") is False

    group = _load_group()  # v0001
    lease = _make_lease(group).model_copy(update={"policy_version": "v_other"})
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(
        group, lease, manifest, now=datetime(2026, 5, 10, 0, 0, 30, tzinfo=timezone.utc)
    )
    assert decision is not None
    assert decision.rejection_reason == RejectionReason.POLICY_VERSION_MISMATCH


def test_known_versions_ordered_by_first_seen():
    tracker = PoolReloadTracker()
    tracker.record_ack(_ack("a", "v1"))
    tracker.record_ack(_ack("b", "v2"))
    tracker.record_ack(_ack("c", "v1"))
    versions = tracker.known_versions()
    assert versions == ["v1", "v2"] or versions == ["v2", "v1"]
    assert "v1" in versions and "v2" in versions
