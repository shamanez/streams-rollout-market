"""LiveStore — in-memory reference implementation of the rollout lifecycle.

A LiveStore receives SampleGroups paired with their pre-OPBC GroupDecision
and tracks each group through the {ACCEPTED, CORRECTABLE, QUARANTINED,
REJECTED} state machine. The store is the source of truth for downstream
consumers (trainer clients) and is the place where idempotency lives:
duplicate group_id submissions are refused, so a retried worker submission
cannot create two records for the same logical group.

Defaults are conservative: get_batch serves only ACCEPTED and CORRECTABLE
groups. QUARANTINED groups are retained (so they can be inspected and,
when an experimental contract allows it, opted into) but never served by
default. REJECTED groups are retained for audit but never served.

Production deployments should swap the storage layer for Postgres / Redis
/ Kafka / object storage — the contract surface here is what they need to
mirror.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .contracts import GroupDecision, GroupStatus, RejectionReason, SampleGroup


class DuplicateGroupError(ValueError):
    """Raised when a SampleGroup with an already-stored group_id is submitted."""


class UnknownGroupError(KeyError):
    """Raised when an operation references a group_id the store has not seen."""


_SERVABLE_STATES: frozenset[GroupStatus] = frozenset(
    {GroupStatus.ACCEPTED, GroupStatus.CORRECTABLE}
)


@dataclass
class StoredGroup:
    """One LiveStore record: the group, its current state, and provenance."""

    group: SampleGroup
    state: GroupStatus
    reason: str
    rejection_reason: RejectionReason | None = None
    served_count: int = 0
    stored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_state_change_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def group_id(self) -> str:
        return self.group.group_id


class InMemoryLiveStore:
    """In-memory LiveStore with insertion-ordered iteration.

    Insertion order is preserved so get_batch is FIFO across servable
    states. The same instance is safe to use across the registry, the
    dispatcher, and the trainer client in a single-process demo; for
    multi-process deployments, swap the storage layer.
    """

    def __init__(self) -> None:
        self._records: OrderedDict[str, StoredGroup] = OrderedDict()

    def submit(self, group: SampleGroup, decision: GroupDecision) -> StoredGroup:
        """Store one group with its decision. Idempotent on duplicate group_id by *raising*.

        Returns the newly stored record. The caller is responsible for
        catching DuplicateGroupError if it intends to treat retries as
        idempotent. We do not silently overwrite because a second submission
        with the same id but different content is exactly the case we must
        not let through.
        """
        if decision.group_id != group.group_id:
            raise ValueError(
                f"decision.group_id {decision.group_id!r} != group.group_id {group.group_id!r}"
            )
        if group.group_id in self._records:
            raise DuplicateGroupError(
                f"group_id {group.group_id!r} already in LiveStore"
            )
        record = StoredGroup(
            group=group,
            state=decision.status,
            reason=decision.reason,
            rejection_reason=decision.rejection_reason,
        )
        self._records[group.group_id] = record
        return record

    def get(self, group_id: str) -> StoredGroup:
        """Return the StoredGroup for group_id, or raise UnknownGroupError."""
        try:
            return self._records[group_id]
        except KeyError as exc:
            raise UnknownGroupError(group_id) from exc

    def has(self, group_id: str) -> bool:
        return group_id in self._records

    def transition(
        self,
        group_id: str,
        new_state: GroupStatus,
        reason: str = "",
        *,
        rejection_reason: RejectionReason | None = None,
    ) -> StoredGroup:
        """Move a stored group to a new state.

        The contract enforces that REJECTED carries a typed rejection_reason
        and that no other state does — same invariant GroupDecision uses.
        Transitions out of REJECTED are not allowed: a rejection is final.
        """
        record = self.get(group_id)
        if record.state == GroupStatus.REJECTED and new_state != GroupStatus.REJECTED:
            raise ValueError(f"group {group_id!r} is REJECTED; transitions are not allowed")
        if new_state == GroupStatus.REJECTED and rejection_reason is None:
            raise ValueError("transition to REJECTED requires a rejection_reason")
        if new_state != GroupStatus.REJECTED and rejection_reason is not None:
            raise ValueError("rejection_reason is only valid when new_state is REJECTED")
        record.state = new_state
        record.reason = reason
        record.rejection_reason = rejection_reason
        record.last_state_change_at = datetime.now(timezone.utc)
        return record

    def get_batch(
        self,
        max_groups: int,
        *,
        include_quarantined: bool = False,
    ) -> list[StoredGroup]:
        """Return up to max_groups servable records in insertion order.

        Default servable states are ACCEPTED and CORRECTABLE. Setting
        include_quarantined=True opts the caller into QUARANTINED groups —
        useful for replay-tier experiments. REJECTED groups are never
        served. Returned records have their served_count incremented; the
        records themselves stay in the store so trainer-side filters can
        re-fetch by group_id if needed.
        """
        if max_groups <= 0:
            return []
        servable = set(_SERVABLE_STATES)
        if include_quarantined:
            servable.add(GroupStatus.QUARANTINED)
        batch: list[StoredGroup] = []
        for record in self._records.values():
            if record.state not in servable:
                continue
            batch.append(record)
            record.served_count += 1
            if len(batch) >= max_groups:
                break
        return batch

    def count_by_state(self) -> dict[GroupStatus, int]:
        """Return the count of stored records grouped by state."""
        counts: dict[GroupStatus, int] = {state: 0 for state in GroupStatus}
        for record in self._records.values():
            counts[record.state] += 1
        return counts

    def __len__(self) -> int:
        """Total number of stored records, regardless of state."""
        return len(self._records)
