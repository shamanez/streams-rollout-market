"""Local worker broker — registry, heartbeats, leases, failures, audit trail.

Mediates between a LiveStore and a fleet of WorkerSDK clients in a single
process. The broker is thread-safe enough for the local demo workload
(many workers, one process) but is not a distributed scheduler — for
multi-process deployments, swap the storage layer for a real queue.

The broker is firewall-clean: it never inspects, replays, or rewrites the
SampleGroup payload. It only matches RolloutJobs to registered workers,
tracks lease state, and records an audit trail of lifecycle events so a
crashed worker (or spot interruption) cannot corrupt group state without
leaving a trace.
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


class WorkerHealth(str, Enum):
    """Heartbeat-derived health classification.

    ACTIVE: heartbeat within heartbeat_ttl_s.
    STALE: heartbeat older than ttl but worker not yet declared dead.
    DEAD: worker explicitly abandoned via abandon_stale_workers.
    UNKNOWN: registered but never heartbeated.
    """

    ACTIVE = "active"
    STALE = "stale"
    DEAD = "dead"
    UNKNOWN = "unknown"


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


@dataclass(frozen=True)
class AuditEntry:
    """One lifecycle event in the broker's append-only audit log."""

    timestamp: datetime
    event: str
    worker_id: str | None
    lease_id: str | None
    job_id: str | None
    detail: str = ""

    def as_dict(self) -> dict[str, str | None]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "event": self.event,
            "worker_id": self.worker_id,
            "lease_id": self.lease_id,
            "job_id": self.job_id,
            "detail": self.detail,
        }


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
    """In-process broker with thread-safe registries, lease state, and audit log.

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
        heartbeat_ttl_s: float = 30.0,
    ) -> None:
        self._store = store
        self._lease_ttl_s = lease_ttl_s
        self._heartbeat_ttl_s = heartbeat_ttl_s
        self._lock = threading.RLock()
        self._workers: dict[str, WorkerManifest] = {}
        self._heartbeats: dict[str, WorkerHeartbeat] = {}
        self._dead_workers: set[str] = set()
        self._jobs: OrderedDict[str, RolloutJob] = OrderedDict()
        self._leased_job_ids: set[str] = set()
        self._leases: dict[str, LeaseRecord] = {}
        self._idempotency_keys: dict[str, str] = {}
        self._failures: list[dict[str, str]] = []
        self._audit: list[AuditEntry] = []

    def _record(
        self,
        event: str,
        *,
        worker_id: str | None = None,
        lease_id: str | None = None,
        job_id: str | None = None,
        detail: str = "",
        when: datetime | None = None,
    ) -> None:
        self._audit.append(
            AuditEntry(
                timestamp=when or datetime.now(timezone.utc),
                event=event,
                worker_id=worker_id,
                lease_id=lease_id,
                job_id=job_id,
                detail=detail,
            )
        )

    def register_worker(self, manifest: WorkerManifest) -> None:
        """Record a worker's static capabilities. Re-registering replaces in place."""
        with self._lock:
            self._workers[manifest.worker_id] = manifest
            self._dead_workers.discard(manifest.worker_id)
            self._record("worker_registered", worker_id=manifest.worker_id)

    def heartbeat(self, beat: WorkerHeartbeat) -> None:
        """Record a heartbeat. The worker must be registered first.

        A heartbeat from a worker previously declared DEAD revives it: the
        worker either recovered from a transient outage, or a spot
        instance came back. Either way, refusing the heartbeat would be
        worse than re-admitting the worker — old leases stay ABANDONED.
        """
        with self._lock:
            if beat.worker_id not in self._workers:
                raise WorkerNotRegisteredError(beat.worker_id)
            revived = beat.worker_id in self._dead_workers
            self._heartbeats[beat.worker_id] = beat
            self._dead_workers.discard(beat.worker_id)
            if revived:
                self._record("worker_revived", worker_id=beat.worker_id)

    def submit_job(self, job: RolloutJob) -> None:
        """Enqueue a RolloutJob for any eligible registered worker."""
        with self._lock:
            self._jobs[job.job_id] = job
            self._record("job_enqueued", job_id=job.job_id)

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
        """Issue a lease to ``worker_id`` for the next eligible queued job."""
        with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotRegisteredError(worker_id)
            if worker_id in self._dead_workers:
                raise WorkerNotRegisteredError(
                    f"worker {worker_id!r} is marked DEAD; heartbeat to revive"
                )
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
                self._record(
                    "lease_issued",
                    worker_id=worker_id,
                    lease_id=lease.lease_id,
                    job_id=job_id,
                    when=ref_now,
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
        """Persist a finished group and close the lease."""
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
            self._record(
                "lease_submitted",
                worker_id=worker_id,
                lease_id=lease_id,
                job_id=record.lease.job_id,
                detail=group.group_id,
            )
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
            self._record(
                "lease_failed",
                worker_id=worker_id,
                lease_id=lease_id,
                job_id=record.lease.job_id,
                detail=f"{reason}: {detail or ''}",
            )

    def reap_expired(self, now: datetime | None = None) -> list[str]:
        """Mark issued leases past their TTL as EXPIRED; return their lease ids.

        EXPIRED is distinct from ABANDONED: a lease can expire because the
        worker is busy or slow on a single rollout, even when the worker is
        otherwise heartbeating. Worker death is reported via
        abandon_stale_workers().
        """
        ref_now = now or datetime.now(timezone.utc)
        expired: list[str] = []
        with self._lock:
            for lease_id, record in self._leases.items():
                if record.state == LeaseState.ISSUED and record.lease.is_expired(ref_now):
                    record.state = LeaseState.EXPIRED
                    record.closed_at = ref_now
                    self._leased_job_ids.discard(record.lease.job_id)
                    expired.append(lease_id)
                    self._record(
                        "lease_expired",
                        worker_id=record.lease.worker_id,
                        lease_id=lease_id,
                        job_id=record.lease.job_id,
                        when=ref_now,
                    )
        return expired

    def worker_health(
        self, worker_id: str, *, now: datetime | None = None
    ) -> WorkerHealth:
        """Return the heartbeat-derived health of one worker."""
        with self._lock:
            if worker_id not in self._workers:
                raise WorkerNotRegisteredError(worker_id)
            if worker_id in self._dead_workers:
                return WorkerHealth.DEAD
            beat = self._heartbeats.get(worker_id)
            if beat is None:
                return WorkerHealth.UNKNOWN
            ref_now = now or datetime.now(timezone.utc)
            age = (ref_now - beat.last_seen_at).total_seconds()
            if age > self._heartbeat_ttl_s:
                return WorkerHealth.STALE
            return WorkerHealth.ACTIVE

    def abandon_stale_workers(
        self, *, now: datetime | None = None
    ) -> dict[str, list[str]]:
        """Mark workers whose heartbeats are stale as DEAD and abandon their leases.

        Returns ``{worker_id: [lease_id, ...]}`` — the workers newly marked
        dead and the lease ids that were ABANDONED on their behalf. Jobs
        held by abandoned leases are returned to the queue so a fresh
        worker can pick them up. Each abandonment is recorded in the audit
        log so the trail is preserved even after the lease record is
        garbage collected.
        """
        ref_now = now or datetime.now(timezone.utc)
        abandoned: dict[str, list[str]] = {}
        with self._lock:
            for worker_id, manifest in self._workers.items():
                _ = manifest  # registered set; iteration over manifests is intentional
                if worker_id in self._dead_workers:
                    continue
                beat = self._heartbeats.get(worker_id)
                if beat is None:
                    # Registered but never heartbeated: only declare dead if the
                    # registry has held the worker longer than the heartbeat TTL.
                    continue
                age = (ref_now - beat.last_seen_at).total_seconds()
                if age <= self._heartbeat_ttl_s:
                    continue
                self._dead_workers.add(worker_id)
                lease_ids: list[str] = []
                for lease_id, record in self._leases.items():
                    if record.lease.worker_id != worker_id:
                        continue
                    if record.state != LeaseState.ISSUED:
                        continue
                    record.state = LeaseState.ABANDONED
                    record.failure_reason = "worker_dead"
                    record.failure_detail = f"heartbeat age {age:.1f}s exceeds ttl"
                    record.closed_at = ref_now
                    self._leased_job_ids.discard(record.lease.job_id)
                    lease_ids.append(lease_id)
                    self._record(
                        "lease_abandoned",
                        worker_id=worker_id,
                        lease_id=lease_id,
                        job_id=record.lease.job_id,
                        detail=f"heartbeat_age_s={age:.1f}",
                        when=ref_now,
                    )
                self._record(
                    "worker_marked_dead",
                    worker_id=worker_id,
                    detail=f"heartbeat_age_s={age:.1f}",
                    when=ref_now,
                )
                abandoned[worker_id] = lease_ids
        return abandoned

    def lease(self, lease_id: str) -> LeaseRecord:
        with self._lock:
            return self._leases[lease_id]

    def heartbeats(self) -> Iterable[WorkerHeartbeat]:
        with self._lock:
            return list(self._heartbeats.values())

    def failures(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self._failures)

    def audit_log(self) -> list[AuditEntry]:
        """Return a snapshot of the append-only audit log."""
        with self._lock:
            return list(self._audit)

    def is_dead(self, worker_id: str) -> bool:
        with self._lock:
            return worker_id in self._dead_workers
