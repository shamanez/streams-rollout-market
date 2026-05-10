"""Trainer-facing consumer API.

A TrainerClient is the only sanctioned way for a downstream trainer to pull
groups out of a LiveStore. The client is policy-aware: every fetch is
parameterised by the policy version the caller is currently training, an
optional staleness budget, an optional precision_class filter, and a
replay_tier that controls whether QUARANTINED groups (replay-tier only) are
admitted.

This module is firewall-clean — it does not import or implement any loss,
optimizer, gradient step, FSDP/Megatron primitive, or reward-model logic.
It only filters and hands out StoredGroup records by metadata; what the
trainer does with them is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .contracts import GroupStatus, SampleGroup
from .livestore import InMemoryLiveStore, StoredGroup


class ReplayTier(str, Enum):
    """Which servable states a fetch is willing to admit.

    FRESH: only ACCEPTED + CORRECTABLE groups (the default trainer diet).
    REPLAY: also includes QUARANTINED groups, marked as opt-in replay.
    """

    FRESH = "fresh"
    REPLAY = "replay"


@dataclass(frozen=True)
class TrainerBatch:
    """One pull from the LiveStore plus the filter that produced it.

    Carrying the filter alongside the groups makes audit logs self-describing:
    a trainer never has to remember which staleness budget it was running
    under when a problematic group was consumed.
    """

    requested_policy_version: str
    max_staleness: int | None
    precision_class: str | None
    replay_tier: ReplayTier
    groups: list[StoredGroup]

    @property
    def size(self) -> int:
        return len(self.groups)

    def metadata(self) -> list[dict[str, object]]:
        """Lightweight per-group metadata for logging — never includes tokens."""
        rows: list[dict[str, object]] = []
        for record in self.groups:
            g: SampleGroup = record.group
            rows.append(
                {
                    "group_id": g.group_id,
                    "policy_version": g.policy_version,
                    "policy_lag_steps": g.policy_lag_steps,
                    "precision_class": g.precision_class,
                    "quantization_class": g.quantization_class,
                    "tokenizer_hash": g.tokenizer_hash,
                    "checkpoint_digest": g.checkpoint_digest,
                    "engine_name": g.engine_name,
                    "state": record.state.value,
                    "reason": record.reason,
                    "served_count": record.served_count,
                }
            )
        return rows


class TrainerClient:
    """Read-only consumer over a LiveStore. Never mutates group payloads.

    fetch() applies filters in this order: replay_tier (which servable
    states are admitted), policy_version, max_staleness, precision_class.
    Returned groups are StoredGroup records — the caller can read the
    underlying SampleGroup but is expected to treat it as immutable.
    """

    def __init__(self, store: InMemoryLiveStore) -> None:
        self._store = store

    def fetch(
        self,
        *,
        policy_version: str,
        max_groups: int,
        max_staleness: int | None = None,
        precision_class: str | None = None,
        replay_tier: ReplayTier = ReplayTier.FRESH,
    ) -> TrainerBatch:
        """Return up to max_groups records that satisfy the filter.

        - policy_version is required: a trainer always knows which policy
          snapshot it is currently consuming, and groups from a different
          version are never silently swept in.
        - max_staleness is in policy_lag_steps; None means "no limit".
        - precision_class None means "any precision the store contains".
        - replay_tier=FRESH excludes QUARANTINED. REPLAY admits them.
        """
        if max_groups <= 0:
            return TrainerBatch(
                requested_policy_version=policy_version,
                max_staleness=max_staleness,
                precision_class=precision_class,
                replay_tier=replay_tier,
                groups=[],
            )

        include_quarantined = replay_tier == ReplayTier.REPLAY
        # Pull the full servable window without consuming served_count yet —
        # we re-implement the filter here so the trainer client owns its
        # own filtering semantics and the store stays a thin lifecycle layer.
        candidates: list[StoredGroup] = []
        for record in self._store._records.values():  # noqa: SLF001 — intentional
            if record.state == GroupStatus.REJECTED:
                continue
            if record.state == GroupStatus.QUARANTINED and not include_quarantined:
                continue
            if record.state not in {
                GroupStatus.ACCEPTED,
                GroupStatus.CORRECTABLE,
                GroupStatus.QUARANTINED,
            }:
                continue
            g = record.group
            if g.policy_version != policy_version:
                continue
            if max_staleness is not None and g.policy_lag_steps > max_staleness:
                continue
            if precision_class is not None and g.precision_class != precision_class:
                continue
            candidates.append(record)
            if len(candidates) >= max_groups:
                break

        for record in candidates:
            record.served_count += 1

        return TrainerBatch(
            requested_policy_version=policy_version,
            max_staleness=max_staleness,
            precision_class=precision_class,
            replay_tier=replay_tier,
            groups=candidates,
        )
