"""Three-worker concurrent demo.

Spins up three WorkerSDK clients in three threads against one broker /
LiveStore, drains a small queue of RolloutJobs, and reports the resulting
LiveStore counts. No real inference: each worker fabricates a SampleGroup
from its lease and submits it.

Phase 3.3 acceptance: three fake workers run concurrently in a local demo.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    RolloutJob,
    RolloutLease,
    SampleGroup,
    Trajectory,
    WorkerManifest,
)
from rollout_market.livestore import InMemoryLiveStore
from rollout_market.worker import WorkerSDK
from rollout_market.worker_broker import LocalWorkerBroker


def _manifest(worker_id: str) -> WorkerManifest:
    return WorkerManifest(
        worker_id=worker_id,
        engine_name="vllm",
        engine_version="0.6.0",
        device_type="L40S",
        device_count=4,
        precision_classes=["bf16"],
        supported_tokenizer_hashes=["tok-qwen3-32b"],
        max_context_tokens=32768,
        max_concurrent_groups=2,
    )


def _job(idx: int) -> RolloutJob:
    return RolloutJob(
        job_id=f"job-{idx:03d}",
        task_id="demo-task",
        prompt_id=f"prompt-{idx:03d}",
        prompt_token_ids=[101, 202, 303],
        policy_version="v0001",
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-qwen3-32b",
        sampling_config_hash="smp-demo",
        sampling_config={"temperature": 0.7, "max_tokens": 64},
        group_size=2,
        required_precision_class="bf16",
        allowed_engines=["vllm"],
        max_context_tokens=4096,
        idempotency_key=f"idem-{idx:03d}",
    )


def _fake_group(lease: RolloutLease) -> SampleGroup:
    trajectories = [
        Trajectory(
            response_token_ids=[10, 11, 12],
            rollout_logprobs=[-0.4, -0.3, -0.2],
            action_mask=[1, 1, 1],
            tool_output_mask=[0, 0, 0],
            reward=1.0,
        )
        for _ in range(lease.group_size)
    ]
    return SampleGroup(
        group_id=f"g-{lease.lease_id}",
        task_id="demo-task",
        prompt_id=lease.job_id,
        worker_id=lease.worker_id,
        assignment_id=lease.assignment_id,
        policy_version=lease.policy_version,
        checkpoint_digest="sha256:demo",
        tokenizer_hash="tok-qwen3-32b",
        sampling_config_hash="smp-demo",
        engine_name="vllm",
        engine_version="0.6.0",
        device_type="L40S",
        precision_class=lease.required_precision_class,
        prompt_token_ids=[101, 202, 303],
        trajectories=trajectories,
        created_at=datetime.now(timezone.utc),
    )


def _accept(group_id: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.ACCEPTED,
        reason="demo",
        recommended_action="train",
    )


def _worker_loop(sdk: WorkerSDK, served: list[str], lock: threading.Lock) -> None:
    sdk.register()
    sdk.heartbeat(current_policy_version="v0001")
    while True:
        lease = sdk.fetch_lease()
        if lease is None:
            return
        group = _fake_group(lease)
        sdk.submit_group(lease, group, _accept(group.group_id))
        with lock:
            served.append(f"{sdk.worker_id}:{group.group_id}")
        time.sleep(0.001)


def main() -> None:
    store = InMemoryLiveStore()
    broker = LocalWorkerBroker(store, lease_ttl_s=30.0)
    for i in range(9):
        broker.submit_job(_job(i))

    sdks = [WorkerSDK(broker, _manifest(f"worker-{i}")) for i in range(3)]
    served: list[str] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=_worker_loop, args=(sdk, served, lock)) for sdk in sdks
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("jobs queued: 9")
    print(f"groups submitted: {len(served)}")
    by_worker: dict[str, int] = {}
    for entry in served:
        wid = entry.split(":", 1)[0]
        by_worker[wid] = by_worker.get(wid, 0) + 1
    for wid in sorted(by_worker):
        print(f"  {wid}: {by_worker[wid]}")
    counts = {s.value: c for s, c in store.count_by_state().items() if c}
    print(f"livestore_by_state: {counts}")


if __name__ == "__main__":
    main()
