"""Worker SDK — thin client over LocalWorkerBroker.

The SDK is the API a real rollout worker should target: register once,
heartbeat periodically, pull a lease, run the rollout out-of-process, and
either submit_group or report_failure when done. The SDK never executes
inference itself; what produces the SampleGroup is left to the caller.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .contracts import (
    GroupDecision,
    RolloutLease,
    SampleGroup,
    WorkerHeartbeat,
    WorkerManifest,
)
from .livestore import StoredGroup
from .worker_broker import LocalWorkerBroker


class WorkerSDK:
    """One worker's view of the broker.

    A worker should hold exactly one SDK instance and reuse it across
    leases. The SDK is intentionally stateless beyond the worker_id /
    manifest pair; lease bookkeeping lives in the broker so a crashed
    worker re-binding to a new SDK can still be reaped on lease expiry.
    """

    def __init__(self, broker: LocalWorkerBroker, manifest: WorkerManifest) -> None:
        self._broker = broker
        self._manifest = manifest

    @property
    def worker_id(self) -> str:
        return self._manifest.worker_id

    def register(self) -> None:
        """Publish the manifest to the broker. Idempotent: re-registers replace."""
        self._broker.register_worker(self._manifest)

    def heartbeat(
        self,
        *,
        healthy: bool = True,
        current_policy_version: str | None = None,
    ) -> WorkerHeartbeat:
        """Send one heartbeat reflecting current liveness and policy state."""
        beat = WorkerHeartbeat(
            worker_id=self._manifest.worker_id,
            engine_name=self._manifest.engine_name,
            engine_version=self._manifest.engine_version,
            device_type=self._manifest.device_type,
            precision_classes=self._manifest.precision_classes,
            max_context_tokens=self._manifest.max_context_tokens,
            price_hint_per_hour=self._manifest.price_hint_per_hour,
            current_policy_version=current_policy_version,
            healthy=healthy,
            last_seen_at=datetime.now(timezone.utc),
        )
        self._broker.heartbeat(beat)
        return beat

    def fetch_lease(self) -> RolloutLease | None:
        """Pull the next eligible lease from the broker. None when queue is empty."""
        return self._broker.acquire_lease(self._manifest.worker_id)

    def submit_group(
        self,
        lease: RolloutLease,
        group: SampleGroup,
        decision: GroupDecision,
    ) -> StoredGroup:
        """Persist a finished group and close the lease."""
        return self._broker.submit_group(
            self._manifest.worker_id, lease.lease_id, group, decision
        )

    def report_failure(
        self,
        lease: RolloutLease,
        reason: str,
        detail: str | None = None,
    ) -> None:
        """Mark the lease as FAILED and free its job back to the queue."""
        self._broker.report_failure(
            self._manifest.worker_id, lease.lease_id, reason, detail
        )
