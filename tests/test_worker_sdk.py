import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    RolloutJob,
    RolloutLease,
    SampleGroup,
    Trajectory,
    WorkerManifest,
)
from rollout_market.livestore import DuplicateGroupError, InMemoryLiveStore
from rollout_market.worker import WorkerSDK
from rollout_market.worker_broker import (
    LeaseConflictError,
    LeaseState,
    LocalWorkerBroker,
    WorkerNotRegisteredError,
)


def _manifest(
    *,
    worker_id: str = "w0",
    engine: str = "vllm",
    precisions: list[str] | None = None,
    quantizations: list[str] | None = None,
    max_context: int = 4096,
) -> WorkerManifest:
    return WorkerManifest(
        worker_id=worker_id,
        engine_name=engine,
        engine_version="0.6.0",
        device_type="L40S",
        device_count=1,
        precision_classes=precisions or ["bf16"],
        quantization_classes=quantizations or [],
        supported_tokenizer_hashes=["tok-x"],
        max_context_tokens=max_context,
        max_concurrent_groups=1,
    )


def _job(
    *,
    job_id: str = "job-1",
    precision: str = "bf16",
    allowed_engines: list[str] | None = None,
    max_context_tokens: int = 1024,
    quant: str | None = None,
    idempotency_key: str = "idem-1",
) -> RolloutJob:
    return RolloutJob(
        job_id=job_id,
        task_id="t",
        prompt_id="p",
        prompt_token_ids=[1, 2, 3],
        policy_version="v0001",
        checkpoint_digest="sha256:c",
        tokenizer_hash="tok-x",
        sampling_config_hash="smp",
        sampling_config={},
        group_size=2,
        required_precision_class=precision,
        required_quantization_class=quant,
        allowed_engines=allowed_engines or [],
        max_context_tokens=max_context_tokens,
        idempotency_key=idempotency_key,
    )


def _fake_group(lease: RolloutLease, *, group_id: str | None = None) -> SampleGroup:
    return SampleGroup(
        group_id=group_id or f"g-{lease.lease_id}",
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
            for _ in range(lease.group_size)
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


def test_register_then_heartbeat():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    sdk.heartbeat(current_policy_version="v0001")
    assert any(b.worker_id == "w0" for b in broker.heartbeats())


def test_heartbeat_rejected_for_unregistered_worker():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    with pytest.raises(WorkerNotRegisteredError):
        sdk.heartbeat()


def test_fetch_lease_returns_none_when_no_jobs():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    assert sdk.fetch_lease() is None


def test_lease_skipped_when_capabilities_dont_match():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest(precisions=["bf16"]))
    sdk.register()
    broker.submit_job(_job(precision="fp16"))
    assert sdk.fetch_lease() is None


def test_lease_skipped_when_engine_not_allowed():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest(engine="vllm"))
    sdk.register()
    broker.submit_job(_job(allowed_engines=["sglang"]))
    assert sdk.fetch_lease() is None


def test_lease_issued_then_submit_persists_group():
    store, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    broker.submit_job(_job())
    lease = sdk.fetch_lease()
    assert lease is not None
    group = _fake_group(lease)
    sdk.submit_group(lease, group, _accept(group.group_id))
    assert store.has(group.group_id)
    assert broker.lease(lease.lease_id).state == LeaseState.SUBMITTED


def test_two_workers_share_disjoint_jobs():
    _, broker = _make()
    sdk_a = WorkerSDK(broker, _manifest(worker_id="wa"))
    sdk_b = WorkerSDK(broker, _manifest(worker_id="wb"))
    sdk_a.register()
    sdk_b.register()
    broker.submit_job(_job(job_id="j1", idempotency_key="idem-j1"))
    broker.submit_job(_job(job_id="j2", idempotency_key="idem-j2"))
    lease_a = sdk_a.fetch_lease()
    lease_b = sdk_b.fetch_lease()
    assert lease_a is not None and lease_b is not None
    assert lease_a.job_id != lease_b.job_id


def test_submit_with_wrong_worker_id_raises():
    _, broker = _make()
    sdk_a = WorkerSDK(broker, _manifest(worker_id="wa"))
    sdk_b = WorkerSDK(broker, _manifest(worker_id="wb"))
    sdk_a.register()
    sdk_b.register()
    broker.submit_job(_job())
    lease = sdk_a.fetch_lease()
    assert lease is not None
    group = _fake_group(lease)
    with pytest.raises(LeaseConflictError):
        sdk_b.submit_group(lease, group, _accept(group.group_id))


def test_idempotency_key_prevents_duplicate_submission():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    broker.submit_job(_job(job_id="j1", idempotency_key="idem-shared"))
    broker.submit_job(_job(job_id="j2", idempotency_key="idem-shared"))
    lease1 = sdk.fetch_lease()
    assert lease1 is not None
    group1 = _fake_group(lease1, group_id="g1")
    sdk.submit_group(lease1, group1, _accept(group1.group_id))
    lease2 = sdk.fetch_lease()
    assert lease2 is not None
    group2 = _fake_group(lease2, group_id="g2")
    with pytest.raises(LeaseConflictError):
        sdk.submit_group(lease2, group2, _accept(group2.group_id))


def test_duplicate_group_id_rejected_by_store():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    broker.submit_job(_job(job_id="j1", idempotency_key="idem-1"))
    broker.submit_job(_job(job_id="j2", idempotency_key="idem-2"))
    lease1 = sdk.fetch_lease()
    lease2 = sdk.fetch_lease()
    assert lease1 is not None and lease2 is not None
    g1 = _fake_group(lease1, group_id="same-id")
    g2 = _fake_group(lease2, group_id="same-id")
    sdk.submit_group(lease1, g1, _accept("same-id"))
    with pytest.raises(DuplicateGroupError):
        sdk.submit_group(lease2, g2, _accept("same-id"))


def test_report_failure_frees_job_back_to_queue():
    _, broker = _make()
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    broker.submit_job(_job(job_id="j1", idempotency_key="idem-1"))
    lease = sdk.fetch_lease()
    assert lease is not None
    sdk.report_failure(lease, reason="oom", detail="cuda oom at decode")
    assert broker.lease(lease.lease_id).state == LeaseState.FAILED
    failures = broker.failures()
    assert failures and failures[0]["reason"] == "oom"
    release = sdk.fetch_lease()
    assert release is not None
    assert release.job_id == "j1"
    assert release.lease_id != lease.lease_id


def test_reap_expired_marks_old_leases():
    _, broker = _make({"lease_ttl_s": 0.001})
    sdk = WorkerSDK(broker, _manifest())
    sdk.register()
    broker.submit_job(_job(job_id="j1", idempotency_key="idem-1"))
    issued = datetime(2026, 5, 10, 12, 0, 0, tzinfo=timezone.utc)
    lease = broker.acquire_lease("w0", now=issued)
    assert lease is not None
    expired = broker.reap_expired(now=issued + timedelta(minutes=1))
    assert lease.lease_id in expired
    assert broker.lease(lease.lease_id).state == LeaseState.EXPIRED


def test_three_workers_concurrent_drain():
    _, broker = _make()
    sdks = [WorkerSDK(broker, _manifest(worker_id=f"w{i}")) for i in range(3)]
    for sdk in sdks:
        sdk.register()
    for i in range(15):
        broker.submit_job(_job(job_id=f"j{i}", idempotency_key=f"idem-{i}"))

    served: list[str] = []
    lock = threading.Lock()

    def loop(sdk: WorkerSDK) -> None:
        while True:
            lease = sdk.fetch_lease()
            if lease is None:
                return
            group = _fake_group(lease)
            sdk.submit_group(lease, group, _accept(group.group_id))
            with lock:
                served.append(group.group_id)
            time.sleep(0.001)

    threads = [threading.Thread(target=loop, args=(sdk,)) for sdk in sdks]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(served) == 15
    assert len(set(served)) == 15
    counts = {s.value: c for s, c in broker._store.count_by_state().items() if c}  # noqa: SLF001
    assert counts == {"accepted": 15}
