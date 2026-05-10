"""Local worker broker — registry, heartbeats, leases, failures.

Mediates between a LiveStore and a fleet of WorkerSDK clients in a single
process. The broker is thread-safe enough for the local demo workload
(many workers, one process) but is not a distributed scheduler — for
multi-process deployments, swap the storage layer for a real queue.

The broker is firewall-clean: it never inspects, replays, or rewrites the
SampleGroup payload. It only matches RolloutJobs to registered workers
and tracks lease state.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Iterable
from uuid import uuid4

from .contracts import (
    GroupDecision,
    RolloutJob,
    RolloutLease,
    SampleGroup,
    WorkerHeartbeat,
    WorkerManifest,
)
from .livestore import InMemoryLiveStore, StoredGroup


class LeaseState(str, Enum):
    ISSUED = "issued"
    SUBMITTED = "submitted"
    EXPIRED = "expired"
    ABANDONED = "abandoned"
    FAILED = "failed"


@dataclass
class LeaseRecord:
    lease: RolloutLease
    job: RolloutJob
    state: LeaseState = LeaseState.ISSUED
    failure_reason: str | None = None
    failure_detail: str | None = None
    submitted_group_id: str | None = None
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at: datetime | None = None


class WorkerNotRegisteredError(KeyError):
    """Raised when an SDK call references a worker the broker has not seen."""


class NoEligibleWorkerError(RuntimeError):
    """Raised when a job is enqueued but no registered worker matches it."""


class LeaseConflictError(RuntimeError):
    """Raised when a worker tries to claim a job already leased to another worker."""


def _capabilities_match(job: RolloutJob, manifest: WorkerManifest) -> bool:
    if job.required_precision_class not in manifest.precision_classes:
        return False
    if job.max_context_tokens > manifest.max_context_tokens:
        return False
    if (
        job.required_quantization_class
        and job.required_quantization_class not in manifest.quantization_classes
    ):
        return False
    if job.allowed_engines and manifest.engine_name not in job.allowed_engines:
        return False
    return True


class LocalWorkerBroker:
    """In-process broker with thread-safe registries and lease state.

    The lock is deliberately coarse: every public method takes the broker
    lock for the duration of its critical section. This is fine for local
    demos with a handful of workers; production deployments should split
    the storage layer per concern.
    """

    def __init__(
        self,
        store: InMemoryLiveStore,
        *,
        lease_ttl_s: float = 60.0,
    ) -> None:
        self._store = store
        self._lease_ttl_s = lease_ttl_s
        self._lock = threading.RLock()
        self._workers: dict[str, WorkerManifest] = {}
        self._heartbeats: dict[str, WorkerHeartbeat] = {}
        self._jobs: OrderedDict[str, RolloutJob] = OrderedDict()
        self._leased_job_ids: set[str] = set()
        self._leases: dict[str, LeaseRecord] = {}
        self._idempotency_keys: dict[str, str] = {}
        self._failures: list[dict[str, str]] = []

    def register_worker(self, manifest: WorkerManifest) -> None:
        """Record a worker's static capabilities. Re-registering replaces in place."""
        with self._lock:
            self._workers[manifest.worker_id] = manifest

    def heartbeat(self, beat: WorkerHeartbeat) -> None:
        """Record a heartbeat. The worker must be registered first."""
        with self._lock:
            if beat.worker_id not in self._workers:
                raise WorkerNotRegisteredError(beat.worker_id)
            self._heartbeats[beat.worker_id] = beat

    def submit_job(self, job: RolloutJob) -> None:
        """Enqueue a RolloutJob for any eligible registered worker."""
        with self._lock:
            self._jobs[job.job_id] = job

    def workers(self) -> list[WorkerManifest]:
        with self._lock:
            return list(self._workers.values())

    def known_jobs(self) -> list[RolloutJob]:
        with self._lock:
            return list(self._jobs.values())

    def acquire_lease(
        self,
        worker_id: str,
        *,
        now: datetime | None = None,
    ) -> RolloutLease | None:
        """Issue a lease to ``worker_id`` for the next eligible queued job.

        Returns None when no queued job matches the worker's manifest. The
        first matching job is dequeued from the candidate set; subsequent
        calls from the same worker will see a fresh job until the queue
        empties.
        """
        with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotRegisteredError(worker_id)
            manifest = self._workers[worker_id]
            ref_now = now or datetime.now(timezone.utc)
            for job_id, job in self._jobs.items():
                if job_id in self._leased_job_ids:
                    continue
                if not _capabilities_match(job, manifest):
                    continue
                lease = RolloutLease(
                    lease_id=f"lease-{uuid4()}",
                    job_id=job_id,
                    assignment_id=f"assign-{uuid4()}",
                    worker_id=worker_id,
                    policy_version=job.policy_version,
                    required_precision_class=job.required_precision_class,
                    required_quantization_class=job.required_quantization_class,
                    max_context_tokens=job.max_context_tokens,
                    group_size=job.group_size,
                    idempotency_key=job.idempotency_key,
                    issued_at=ref_now,
                    expires_at=ref_now + timedelta(seconds=self._lease_ttl_s),
                )
                self._leased_job_ids.add(job_id)
                self._leases[lease.lease_id] = LeaseRecord(
                    lease=lease, job=job, issued_at=ref_now
                )
                return lease
            return None

    def submit_group(
        self,
        worker_id: str,
        lease_id: str,
        group: SampleGroup,
        decision: GroupDecision,
    ) -> StoredGroup:
        """Persist a finished group and close the lease.

        Idempotency: a second submission with the same idempotency key is
        rejected with LeaseConflictError, so retried workers cannot create
        duplicate StoredGroup records.
        """
        with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotRegisteredError(worker_id)
            try:
                record = self._leases[lease_id]
            except KeyError as exc:
                raise LeaseConflictError(f"unknown lease {lease_id!r}") from exc
            if record.lease.worker_id != worker_id:
                raise LeaseConflictError(
                    f"lease {lease_id!r} is held by {record.lease.worker_id!r}, "
                    f"not {worker_id!r}"
                )
            if record.state != LeaseState.ISSUED:
                raise LeaseConflictError(
                    f"lease {lease_id!r} is in state {record.state.value}, not issued"
                )
            seen_for = self._idempotency_keys.get(record.lease.idempotency_key)
            if seen_for is not None and seen_for != group.group_id:
                raise LeaseConflictError(
                    f"idempotency_key {record.lease.idempotency_key!r} already bound "
                    f"to group {seen_for!r}"
                )
            stored = self._store.submit(group, decision)
            self._idempotency_keys[record.lease.idempotency_key] = group.group_id
            record.state = LeaseState.SUBMITTED
            record.submitted_group_id = group.group_id
            record.closed_at = datetime.now(timezone.utc)
            return stored

    def report_failure(
        self,
        worker_id: str,
        lease_id: str,
        reason: str,
        detail: str | None = None,
    ) -> None:
        """Record a worker-side failure for one lease and free the job back to the queue."""
        with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotRegisteredError(worker_id)
            record = self._leases.get(lease_id)
            if record is None or record.lease.worker_id != worker_id:
                raise LeaseConflictError(f"unknown or unowned lease {lease_id!r}")
            if record.state != LeaseState.ISSUED:
                raise LeaseConflictError(
                    f"lease {lease_id!r} cannot be failed from state {record.state.value}"
                )
            record.state = LeaseState.FAILED
            record.failure_reason = reason
            record.failure_detail = detail
            record.closed_at = datetime.now(timezone.utc)
            self._leased_job_ids.discard(record.lease.job_id)
            self._failures.append(
                {
                    "worker_id": worker_id,
                    "lease_id": lease_id,
                    "job_id": record.lease.job_id,
                    "reason": reason,
                    "detail": detail or "",
                }
            )

    def reap_expired(self, now: datetime | None = None) -> list[str]:
        """Mark issued leases past their TTL as EXPIRED; return their lease ids."""
        ref_now = now or datetime.now(timezone.utc)
        expired: list[str] = []
        with self._lock:
            for lease_id, record in self._leases.items():
                if record.state == LeaseState.ISSUED and record.lease.is_expired(ref_now):
                    record.state = LeaseState.EXPIRED
                    record.closed_at = ref_now
                    self._leased_job_ids.discard(record.lease.job_id)
                    expired.append(lease_id)
        return expired

    def lease(self, lease_id: str) -> LeaseRecord:
        with self._lock:
            return self._leases[lease_id]

    def heartbeats(self) -> Iterable[WorkerHeartbeat]:
        with self._lock:
            return list(self._heartbeats.values())

    def failures(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._failures)
