"""End-to-end marketplace simulation with honest and toxic workers.

Drives the full stack — worker_broker, validators, opbc, livestore — with
a small population of synthetic workers. Each worker has a behaviour
profile that determines whether the SampleGroup it produces is
contract-clean and whether the trainer-side budget will accept it. The
simulation reports the resulting decision distribution so a human can see
that:

  - validators reject toxic workers with the right RejectionReason,
  - OPBC routes within-budget groups to train, off-budget groups to
    replay or train_with_correction, very-stale groups to quarantine,
  - the broker audit log reflects every transition.

The simulation is firewall-clean: it never computes a loss, never runs
an optimizer, and only uses the public package surface.
"""

from __future__ import annotations

import html as _html
import json
import random
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterable

from ..contracts import (
    GroupDecision,
    PolicyManifest,
    RolloutJob,
    RolloutLease,
    SampleGroup,
    Trajectory,
    WorkerManifest,
)
from ..livestore import InMemoryLiveStore
from ..opbc import BudgetPolicy, compute_budget_report, decide_group
from ..validators import validate_group_against_lease
from ..worker import WorkerSDK
from ..worker_broker import LocalWorkerBroker


class WorkerProfile(str, Enum):
    """Behaviour archetype for one synthetic worker.

    HONEST: contract-clean output with within-budget logprobs.
    NOISY: contract-clean but produces high-clipped budgets occasionally.
    STALE: contract-clean but reports policy_lag_steps above replay threshold.
    TOXIC_TOKENIZER: ships a tokenizer_hash that does not match the manifest.
    TOXIC_PRECISION: ships a precision_class that does not match the lease.
    TOXIC_NO_LOGPROBS: ships zero logprobs.
    TOXIC_VERSION_DRIFT: ships a policy_version that does not match the lease.
    """

    HONEST = "honest"
    NOISY = "noisy"
    STALE = "stale"
    TOXIC_TOKENIZER = "toxic_tokenizer"
    TOXIC_PRECISION = "toxic_precision"
    TOXIC_NO_LOGPROBS = "toxic_no_logprobs"
    TOXIC_VERSION_DRIFT = "toxic_version_drift"


@dataclass
class _SimWorker:
    sdk: WorkerSDK
    profile: WorkerProfile


@dataclass
class SimulationResult:
    """Outcome counters for one simulation run."""

    jobs_total: int
    submissions: int
    by_worker_action: dict[tuple[str, str], int] = field(default_factory=dict)
    by_profile_action: dict[tuple[str, str], int] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    decision_reasons: dict[str, int] = field(default_factory=dict)
    livestore_by_state: dict[str, int] = field(default_factory=dict)
    audit_event_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "jobs_total": self.jobs_total,
            "submissions": self.submissions,
            "by_worker_action": [
                {"worker_id": w, "action": a, "count": c}
                for (w, a), c in sorted(self.by_worker_action.items())
            ],
            "by_profile_action": [
                {"profile": p, "action": a, "count": c}
                for (p, a), c in sorted(self.by_profile_action.items())
            ],
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "decision_reasons": dict(sorted(self.decision_reasons.items())),
            "livestore_by_state": dict(sorted(self.livestore_by_state.items())),
            "audit_event_counts": dict(sorted(self.audit_event_counts.items())),
        }


def _manifest(worker_id: str) -> WorkerManifest:
    return WorkerManifest(
        worker_id=worker_id,
        engine_name="vllm",
        engine_version="0.6.0",
        device_type="L40S",
        device_count=1,
        precision_classes=["bf16"],
        supported_tokenizer_hashes=["tok-honest"],
        max_context_tokens=8192,
        max_concurrent_groups=1,
    )


def _job(idx: int) -> RolloutJob:
    return RolloutJob(
        job_id=f"job-{idx:04d}",
        task_id="sim-task",
        prompt_id=f"prompt-{idx:04d}",
        prompt_token_ids=[101, 202, 303, 404],
        policy_version="v0001",
        checkpoint_digest="sha256:sim-ckpt",
        tokenizer_hash="tok-honest",
        sampling_config_hash="smp-v1",
        sampling_config={"temperature": 0.7, "max_tokens": 64},
        group_size=2,
        required_precision_class="bf16",
        allowed_engines=["vllm"],
        max_context_tokens=4096,
        idempotency_key=f"idem-{idx:04d}",
    )


def _policy_manifest() -> PolicyManifest:
    return PolicyManifest(
        policy_version="v0001",
        checkpoint_digest="sha256:sim-ckpt",
        tokenizer_hash="tok-honest",
        model_config_hash="cfg-sim",
        precision_class="bf16",
    )


def _make_group(
    lease: RolloutLease,
    profile: WorkerProfile,
    rng: random.Random,
) -> SampleGroup:
    base_logprobs = [-0.4, -0.3, -0.5]
    if profile == WorkerProfile.NOISY:
        # Drift between clamp (20) and veto (30): one token clipped, none vetoed.
        # Triggers HIGH_CLIPPED_FRACTION -> train_with_correction.
        rollout_logprobs = [-0.4, -0.3, -25.0]
    elif profile == WorkerProfile.STALE:
        rollout_logprobs = list(base_logprobs)
    elif profile == WorkerProfile.TOXIC_NO_LOGPROBS:
        rollout_logprobs = [0.0, 0.0, 0.0]
    else:
        rollout_logprobs = [
            lp + rng.uniform(-0.02, 0.02) for lp in base_logprobs
        ]

    trajectories = [
        Trajectory(
            response_token_ids=[10, 11, 12],
            rollout_logprobs=rollout_logprobs,
            action_mask=[1, 1, 1],
            tool_output_mask=[0, 0, 0],
            reward=1.0,
        )
        for _ in range(lease.group_size)
    ]

    tokenizer_hash = (
        "tok-INVALID" if profile == WorkerProfile.TOXIC_TOKENIZER else "tok-honest"
    )
    precision = (
        "fp16" if profile == WorkerProfile.TOXIC_PRECISION else lease.required_precision_class
    )
    policy_version = (
        "v9999" if profile == WorkerProfile.TOXIC_VERSION_DRIFT else lease.policy_version
    )
    policy_lag = 12 if profile == WorkerProfile.STALE else 1

    return SampleGroup(
        group_id=f"g-{lease.lease_id}",
        task_id="sim-task",
        prompt_id=lease.job_id,
        worker_id=lease.worker_id,
        assignment_id=lease.assignment_id,
        policy_version=policy_version,
        checkpoint_digest="sha256:sim-ckpt",
        tokenizer_hash=tokenizer_hash,
        sampling_config_hash="smp-v1",
        engine_name="vllm",
        engine_version="0.6.0",
        device_type="L40S",
        precision_class=precision,
        prompt_token_ids=[101, 202, 303, 404],
        trajectories=trajectories,
        policy_lag_steps=policy_lag,
        created_at=datetime.now(timezone.utc),
    )


def _train_logprobs_for(group: SampleGroup) -> list[float]:
    """Synthesize trainer-side logprobs.

    The trainer recomputes logprobs against its *own* current policy, so
    the values are anchored to a fixed baseline regardless of what the
    rollout side reported. That is what makes drift visible: a NOISY
    worker's spike to -50 leaves a corresponding spike in the log_ratio
    while the trainer side stays near -0.4.
    """
    baseline = [-0.395, -0.305, -0.495]
    train: list[float] = []
    for traj in group.trajectories:
        idx = 0
        for action, tool in zip(traj.action_mask, traj.tool_output_mask):
            if action and not tool:
                train.append(baseline[idx % len(baseline)])
                idx += 1
    return train


def _decide_for_clean_group(group: SampleGroup) -> GroupDecision:
    train_logprobs = _train_logprobs_for(group)
    report = compute_budget_report(group, train_logprobs)
    return decide_group(report, policy=BudgetPolicy())


def _build_workers(
    broker: LocalWorkerBroker,
    profiles: list[WorkerProfile],
) -> list[_SimWorker]:
    workers: list[_SimWorker] = []
    for idx, profile in enumerate(profiles):
        manifest = _manifest(f"w{idx:02d}-{profile.value}")
        sdk = WorkerSDK(broker, manifest)
        sdk.register()
        sdk.heartbeat(current_policy_version="v0001")
        workers.append(_SimWorker(sdk=sdk, profile=profile))
    return workers


def _worker_loop(
    worker: _SimWorker,
    manifest_for_validator: PolicyManifest,
    rng: random.Random,
    out: list[tuple[str, WorkerProfile, GroupDecision]],
    lock: threading.Lock,
    barrier: threading.Barrier,
) -> None:
    barrier.wait()  # release all workers simultaneously so leases fan out
    while True:
        lease = worker.sdk.fetch_lease()
        if lease is None:
            return
        group = _make_group(lease, worker.profile, rng)
        rejected = validate_group_against_lease(
            group, lease, manifest_for_validator, now=datetime.now(timezone.utc)
        )
        if rejected is not None:
            decision = rejected
        else:
            decision = _decide_for_clean_group(group)
        worker.sdk.submit_group(lease, group, decision)
        with lock:
            out.append((worker.sdk.worker_id, worker.profile, decision))
        time.sleep(0.001)  # yield GIL so other workers get leases


def run_simulation(
    *,
    profiles: Iterable[WorkerProfile],
    num_jobs: int,
    seed: int = 0,
) -> SimulationResult:
    """Run one marketplace simulation and return aggregated outcome counters."""
    rng = random.Random(seed)
    store = InMemoryLiveStore()
    broker = LocalWorkerBroker(store, lease_ttl_s=30.0, heartbeat_ttl_s=120.0)
    workers = _build_workers(broker, list(profiles))

    for i in range(num_jobs):
        broker.submit_job(_job(i))

    manifest = _policy_manifest()
    submissions: list[tuple[str, WorkerProfile, GroupDecision]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(len(workers))
    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(w, manifest, rng, submissions, lock, barrier),
        )
        for w in workers
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    by_worker_action: Counter[tuple[str, str]] = Counter()
    by_profile_action: Counter[tuple[str, str]] = Counter()
    rejection_reasons: Counter[str] = Counter()
    decision_reasons: Counter[str] = Counter()
    for worker_id, profile, decision in submissions:
        by_worker_action[(worker_id, decision.recommended_action)] += 1
        by_profile_action[(profile.value, decision.recommended_action)] += 1
        if decision.rejection_reason is not None:
            rejection_reasons[decision.rejection_reason.value] += 1
        for r in decision.decision_reasons:
            decision_reasons[r.value] += 1

    livestore_by_state = {
        s.value: c for s, c in store.count_by_state().items() if c > 0
    }
    audit_event_counts: Counter[str] = Counter(e.event for e in broker.audit_log())

    return SimulationResult(
        jobs_total=num_jobs,
        submissions=len(submissions),
        by_worker_action=dict(by_worker_action),
        by_profile_action=dict(by_profile_action),
        rejection_reasons=dict(rejection_reasons),
        decision_reasons=dict(decision_reasons),
        livestore_by_state=livestore_by_state,
        audit_event_counts=dict(audit_event_counts),
    )


def render_html(result: SimulationResult) -> str:
    """Render a self-contained HTML dashboard for one simulation result."""

    def _table(headers: list[str], rows: list[list[object]]) -> str:
        head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
        body = "".join(
            "<tr>"
            + "".join(
                f"<td>{_html.escape(str(c))}</td>" for c in row
            )
            + "</tr>"
            for row in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    profile_action_rows = sorted(
        ((p, a, c) for (p, a), c in result.by_profile_action.items()),
        key=lambda r: (r[0], r[1]),
    )
    worker_action_rows = sorted(
        ((w, a, c) for (w, a), c in result.by_worker_action.items()),
        key=lambda r: (r[0], r[1]),
    )

    summary_rows = [
        ["jobs_total", result.jobs_total],
        ["submissions", result.submissions],
    ]

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Marketplace simulation</title>"
        "<style>"
        "body{font-family:system-ui,sans-serif;max-width:920px;margin:2rem auto;color:#111}"
        "table{border-collapse:collapse;width:100%;margin-bottom:1rem}"
        "th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left}"
        "th{background:#f5f5f5}"
        "h1{font-size:1.2rem}h2{font-size:1rem;margin-top:1.5rem}"
        "</style></head><body>"
        "<h1>Marketplace simulation</h1>"
        "<h2>Run summary</h2>"
        + _table(["metric", "value"], summary_rows)
        + "<h2>Decisions by worker profile</h2>"
        + _table(["profile", "action", "count"], [list(r) for r in profile_action_rows])
        + "<h2>Decisions by worker id</h2>"
        + _table(["worker_id", "action", "count"], [list(r) for r in worker_action_rows])
        + "<h2>Rejection reasons (validators)</h2>"
        + _table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(result.rejection_reasons.items())],
        )
        + "<h2>Decision reasons (OPBC)</h2>"
        + _table(
            ["reason", "count"],
            [[k, v] for k, v in sorted(result.decision_reasons.items())],
        )
        + "<h2>LiveStore by state</h2>"
        + _table(
            ["state", "count"],
            [[k, v] for k, v in sorted(result.livestore_by_state.items())],
        )
        + "<h2>Broker audit-event counts</h2>"
        + _table(
            ["event", "count"],
            [[k, v] for k, v in sorted(result.audit_event_counts.items())],
        )
        + "</body></html>"
    )


def write_simulation(result: SimulationResult, out_dir: Path | str) -> dict[str, Path]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    json_path = out_path / "marketplace_simulation.json"
    html_path = out_path / "marketplace_simulation.html"
    json_path.write_text(json.dumps(result.as_dict(), indent=2), encoding="utf-8")
    html_path.write_text(render_html(result), encoding="utf-8")
    return {"json": json_path, "html": html_path}
