"""Pool reload quorum tracking — k-of-n weight-sync coordination.

When the trainer publishes a new PolicyManifest, every worker in the pool
needs to reload weights before it can serve the new version. Synchronous
all-or-nothing reloads block trainer progress whenever one worker is slow,
restarting, or absent. PoolReloadTracker decouples *pool-level visibility*
of a new version from *strict per-group pinning*: the registry can declare
a version 'pool-active' as soon as k of n workers have acked, while every
SampleGroup still carries the exact policy_version + checkpoint_digest the
worker actually served.

This module owns *only* the quorum bookkeeping. The contract that keeps
mixed-version groups out of training (RejectionReason.MIXED_POLICY_VERSION,
RejectionReason.POLICY_VERSION_MISMATCH) lives in validators.py and is not
loosened by the existence of this tracker.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class ReloadAck:
    """One worker's ack that it has loaded a specific policy version.

    Acks are immutable: a fresh ack for the same (worker_id, version) pair
    overwrites timestamp metadata but does not change quorum membership.
    """

    worker_id: str
    policy_version: str
    checkpoint_digest: str
    acked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class QuorumViolation(ValueError):
    """Raised when a worker reports a checkpoint_digest that disagrees with
    a previous ack from another worker for the same policy_version."""


class PoolReloadTracker:
    """Records per-worker reload acks and computes the pool-active version.

    Invariants:
      - One worker can only be in the quorum for one version at a time.
        A new ack from the same worker replaces any previous ack.
      - Two workers acking the same policy_version with different
        checkpoint_digests is treated as a contract violation and raises
        QuorumViolation rather than silently letting one digest win.
      - pool_active_version(k, n) returns the highest version whose ack
        count is >= k. Older versions can stay 'active' for already-issued
        leases; the per-group pinning contract still applies.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._acks: dict[str, ReloadAck] = {}  # by worker_id
        self._digest_by_version: dict[str, str] = {}

    def record_ack(self, ack: ReloadAck) -> None:
        with self._lock:
            existing_digest = self._digest_by_version.get(ack.policy_version)
            if existing_digest is None:
                self._digest_by_version[ack.policy_version] = ack.checkpoint_digest
            elif existing_digest != ack.checkpoint_digest:
                raise QuorumViolation(
                    f"version {ack.policy_version!r} ack with digest "
                    f"{ack.checkpoint_digest!r} disagrees with previously seen "
                    f"{existing_digest!r}"
                )
            self._acks[ack.worker_id] = ack

    def ack_count(self, policy_version: str) -> int:
        with self._lock:
            return sum(1 for a in self._acks.values() if a.policy_version == policy_version)

    def workers_acked(self, policy_version: str) -> set[str]:
        with self._lock:
            return {
                a.worker_id for a in self._acks.values() if a.policy_version == policy_version
            }

    def known_versions(self) -> list[str]:
        with self._lock:
            seen: list[str] = []
            for ack in self._acks.values():
                if ack.policy_version not in seen:
                    seen.append(ack.policy_version)
            return seen

    def has_quorum(self, policy_version: str, *, k: int, n: int | None = None) -> bool:
        """Return True when the version has at least k acks.

        ``n`` is informational — the caller's notion of the full pool size.
        It is recorded for symmetry with the design doc but does not gate
        the answer; quorum is a function of k and the live ack count.
        """
        if k <= 0:
            raise ValueError("k must be positive")
        if n is not None and n < k:
            raise ValueError(f"n={n} is smaller than k={k}; quorum impossible")
        return self.ack_count(policy_version) >= k

    def pool_active_version(
        self,
        *,
        k: int,
        n: int | None = None,
        candidates: list[str] | None = None,
    ) -> str | None:
        """Return the latest version that has reached k-of-n quorum, or None.

        ``candidates`` orders the candidate versions newest-first; if it is
        omitted, the registry's record of seen versions is used in
        insertion order (oldest-first). Callers that publish strictly
        monotonic versions should pass their own newest-first list to get
        a deterministic answer when several versions hit quorum
        simultaneously.
        """
        if candidates is None:
            candidates = list(reversed(self.known_versions()))
        for version in candidates:
            if self.has_quorum(version, k=k, n=n):
                return version
        return None

    def is_worker_aligned(self, worker_id: str, policy_version: str) -> bool:
        """Return True iff ``worker_id`` has acked exactly ``policy_version``.

        Dispatchers MUST gate per-job dispatch on this — the pool-active
        version is informational only; the canonical safety check is
        whether the *specific* worker in front of the dispatcher has the
        weights for the version the job is pinned to.
        """
        with self._lock:
            ack = self._acks.get(worker_id)
            return ack is not None and ack.policy_version == policy_version

    def __len__(self) -> int:
        with self._lock:
            return len(self._acks)
