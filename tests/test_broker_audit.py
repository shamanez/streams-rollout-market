"""Phase 5.1: heartbeat / lease timeout / audit-trail tests.

Worker death and spot interruption must not corrupt group state. The
broker proves this by:
  - distinguishing EXPIRED (lease TTL elapsed) from ABANDONED (worker dead)
  - returning every freed job to the queue so a fresh worker can pick it up
  - recording every lifecycle transition in an append-only audit log
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    RolloutJob,
    RolloutLease,
    SampleGroup,
    Trajectory,
    WorkerHeartbeat,
    WorkerManifest,
)
from rollout_market.livestore import InMemoryLiveStore
from rollout_market.worker_broker import (
    LeaseState,
    LocalWorkerBroker,
    WorkerHealth,
    WorkerNotRegisteredError,
)


def _manifest(worker_id: str = "w0") -> WorkerManifest:
    return WorkerManifest(
        worker_id=worker_id,
        engine_name="vllm",
        engine_version="0.6.0",
        device_type="L40S",
        device_count=1,
        precision_classes=["bf16"],
        supported_tokenizer_hashes=["tok-x"],
        max_context_tokens=4096,
        max_concurrent_groups=1,
    )


def _job(idx: int = 1) -> RolloutJob:
    return RolloutJob(
        job_id=f"job-{idx}",
        task_id="t",
        prompt_id="p",
        prompt_token_ids=[1, 2, 3],
        policy_version="v0001",
        checkpoint_digest="sha256:c",
        tokenizer_hash="tok-x",
        sampling_config_hash="smp",
        sampling_config={},
        group_size=1,
        required_precision_class="bf16",
        max_context_tokens=1024,
        idempotency_key=f"idem-{idx}",
    )


def _heartbeat(manifest: WorkerManifest, *, when: datetime) -> WorkerHeartbeat:
    return WorkerHeartbeat(
        worker_id=manifest.worker_id,
        engine_name=manifest.engine_name,
        engine_version=manifest.engine_version,
        device_type=manifest.device_type,
        precision_classes=manifest.precision_classes,
        max_context_tokens=manifest.max_context_tokens,
        last_seen_at=when,
    )


def _fake_group(lease: RolloutLease) -> SampleGroup:
    return SampleGroup(
        group_id=f"g-{lease.lease_id}",
        task_id="t",
        prompt_id=lease.job_id,
        worker_id=lease.worker_id,
        assignment_id=lease.assignment_id,
        policy_version=lease.policy_version,
        checkpoint_digest="sha256:c",
        tokenizer_hash="tok-x",
        sampling_config_hash="smp",
        engine_name="vllm",
        engine_version="0.6.0",
        device_type="L40S",
        precision_class=lease.required_precision_class,
        prompt_token_ids=[1, 2, 3],
        trajectories=[
            Trajectory(
                response_token_ids=[10, 11],
                rollout_logprobs=[-0.5, -0.4],
                action_mask=[1, 1],
                tool_output_mask=[0, 0],
                reward=1.0,
            )
        ],
    )


def _accept(group_id: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.ACCEPTED,
        reason="ok",
        recommended_action="train",
    )


def _make(broker_kwargs: dict | None = None) -> tuple[InMemoryLiveStore, LocalWorkerBroker]:
    store = InMemoryLiveStore()
    broker = LocalWorkerBroker(store, **(broker_kwargs or {}))
    return store, broker


# --- worker health classification -------------------------------------------------


def test_worker_unknown_until_first_heartbeat():
    _, broker = _make()
    broker.register_worker(_manifest())
    assert broker.worker_health("w0") == WorkerHealth.UNKNOWN


def test_worker_active_within_ttl():
    _, broker = _make({"heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    assert broker.worker_health("w0", now=t0 + timedelta(seconds=10)) == WorkerHealth.ACTIVE


def test_worker_stale_past_ttl():
    _, broker = _make({"heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    assert broker.worker_health("w0", now=t0 + timedelta(seconds=120)) == WorkerHealth.STALE


def test_worker_health_unknown_worker_raises():
    _, broker = _make()
    with pytest.raises(WorkerNotRegisteredError):
        broker.worker_health("ghost")


# --- abandon_stale_workers --------------------------------------------------------


def test_abandon_stale_workers_marks_dead_and_abandons_lease():
    _, broker = _make({"lease_ttl_s": 600.0, "heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    lease = broker.acquire_lease("w0", now=t0)
    assert lease is not None

    abandoned = broker.abandon_stale_workers(now=t0 + timedelta(seconds=120))
    assert abandoned == {"w0": [lease.lease_id]}
    assert broker.is_dead("w0")
    assert broker.worker_health("w0", now=t0 + timedelta(seconds=120)) == WorkerHealth.DEAD
    assert broker.lease(lease.lease_id).state == LeaseState.ABANDONED


def test_abandoned_job_returns_to_queue_for_fresh_worker():
    _, broker = _make({"lease_ttl_s": 600.0, "heartbeat_ttl_s": 30.0})
    a = _manifest("wa")
    b = _manifest("wb")
    broker.register_worker(a)
    broker.register_worker(b)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(a, when=t0))
    broker.heartbeat(_heartbeat(b, when=t0))
    broker.submit_job(_job(1))
    first = broker.acquire_lease("wa", now=t0)
    assert first is not None

    later = t0 + timedelta(seconds=120)
    broker.heartbeat(_heartbeat(b, when=later))  # b stays alive
    broker.abandon_stale_workers(now=later)

    second = broker.acquire_lease("wb", now=later)
    assert second is not None
    assert second.job_id == first.job_id
    assert second.lease_id != first.lease_id


def test_abandon_stale_does_not_touch_already_submitted_lease():
    store, broker = _make({"lease_ttl_s": 600.0, "heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    lease = broker.acquire_lease("w0", now=t0)
    assert lease is not None
    group = _fake_group(lease)
    broker.submit_group("w0", lease.lease_id, group, _accept(group.group_id))

    broker.abandon_stale_workers(now=t0 + timedelta(seconds=120))
    assert broker.lease(lease.lease_id).state == LeaseState.SUBMITTED
    assert store.has(group.group_id)


def test_dead_worker_cannot_acquire_lease_until_revived():
    _, broker = _make({"heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    broker.abandon_stale_workers(now=t0 + timedelta(seconds=120))

    with pytest.raises(WorkerNotRegisteredError):
        broker.acquire_lease("w0", now=t0 + timedelta(seconds=130))

    broker.heartbeat(_heartbeat(manifest, when=t0 + timedelta(seconds=140)))
    assert not broker.is_dead("w0")
    revived = broker.acquire_lease("w0", now=t0 + timedelta(seconds=141))
    assert revived is not None


# --- expired vs abandoned distinction --------------------------------------------


def test_reap_expired_uses_lease_ttl_not_heartbeat_ttl():
    _, broker = _make({"lease_ttl_s": 5.0, "heartbeat_ttl_s": 600.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    lease = broker.acquire_lease("w0", now=t0)
    assert lease is not None

    expired = broker.reap_expired(now=t0 + timedelta(seconds=10))
    assert expired == [lease.lease_id]
    assert broker.lease(lease.lease_id).state == LeaseState.EXPIRED
    assert not broker.is_dead("w0")  # heartbeat still fresh


def test_audit_distinguishes_expired_and_abandoned():
    _, broker = _make({"lease_ttl_s": 600.0, "heartbeat_ttl_s": 30.0})
    a = _manifest("wa")
    b = _manifest("wb")
    broker.register_worker(a)
    broker.register_worker(b)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(a, when=t0))
    broker.heartbeat(_heartbeat(b, when=t0))
    broker.submit_job(_job(1))
    broker.submit_job(_job(2))
    lease_a = broker.acquire_lease("wa", now=t0)
    lease_b = broker.acquire_lease("wb", now=t0)
    assert lease_a is not None and lease_b is not None

    # wb stays alive; wa goes silent
    later = t0 + timedelta(seconds=120)
    broker.heartbeat(_heartbeat(b, when=later))

    # Force lease_b to also expire (separate from heartbeat death) by
    # reaping with a future timestamp.
    far_future = t0 + timedelta(seconds=10_000)
    broker.heartbeat(_heartbeat(b, when=far_future))
    broker.abandon_stale_workers(now=far_future)
    broker.reap_expired(now=far_future)

    events = {e.event for e in broker.audit_log()}
    assert "lease_abandoned" in events
    assert "lease_expired" in events
    assert "worker_marked_dead" in events


# --- audit log content ------------------------------------------------------------


def test_audit_log_records_full_lifecycle():
    store, broker = _make()
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    lease = broker.acquire_lease("w0", now=t0)
    assert lease is not None
    group = _fake_group(lease)
    broker.submit_group("w0", lease.lease_id, group, _accept(group.group_id))

    events = [e.event for e in broker.audit_log()]
    assert "worker_registered" in events
    assert "job_enqueued" in events
    assert "lease_issued" in events
    assert "lease_submitted" in events


def test_audit_log_records_failure_path():
    _, broker = _make()
    manifest = _manifest()
    broker.register_worker(manifest)
    broker.heartbeat(
        _heartbeat(manifest, when=datetime(2026, 5, 10, tzinfo=timezone.utc))
    )
    broker.submit_job(_job())
    lease = broker.acquire_lease("w0")
    assert lease is not None
    broker.report_failure("w0", lease.lease_id, "oom", "cuda oom at decode")

    failure_entries = [e for e in broker.audit_log() if e.event == "lease_failed"]
    assert len(failure_entries) == 1
    assert "oom" in (failure_entries[0].detail or "")


def test_audit_entry_serializes_to_dict():
    _, broker = _make()
    broker.register_worker(_manifest())
    entry = broker.audit_log()[0]
    payload = entry.as_dict()
    for required in ("timestamp", "event", "worker_id", "lease_id", "job_id", "detail"):
        assert required in payload
    assert payload["event"] == "worker_registered"


def test_repeated_abandon_call_is_noop_for_already_dead_worker():
    _, broker = _make({"heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    broker.acquire_lease("w0", now=t0)
    later = t0 + timedelta(seconds=120)

    first = broker.abandon_stale_workers(now=later)
    second = broker.abandon_stale_workers(now=later)
    assert "w0" in first
    assert second == {}


def test_revived_worker_keeps_old_abandoned_leases_abandoned():
    _, broker = _make({"heartbeat_ttl_s": 30.0})
    manifest = _manifest()
    broker.register_worker(manifest)
    t0 = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    broker.heartbeat(_heartbeat(manifest, when=t0))
    broker.submit_job(_job())
    lease = broker.acquire_lease("w0", now=t0)
    assert lease is not None
    later = t0 + timedelta(seconds=120)
    broker.abandon_stale_workers(now=later)
    broker.heartbeat(_heartbeat(manifest, when=later + timedelta(seconds=1)))

    assert broker.lease(lease.lease_id).state == LeaseState.ABANDONED
    revived_events = [e for e in broker.audit_log() if e.event == "worker_revived"]
    assert len(revived_events) == 1
