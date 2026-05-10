"""Fake trainer demo — pulls groups by metadata and prints them.

Deliberately does NOT compute a loss, run an optimizer, or touch any
trainer primitive. The whole point of this example is to show that the
TrainerClient is firewall-clean: a real trainer wires its loss code on the
*outside* of this module.
"""

from __future__ import annotations

import json
from pathlib import Path

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    RejectionReason,
    SampleGroup,
)
from rollout_market.livestore import InMemoryLiveStore
from rollout_market.trainer_client import ReplayTier, TrainerClient


def _decide(group_id: str, status: GroupStatus, reason: str) -> GroupDecision:
    rec_action = {
        GroupStatus.ACCEPTED: "train",
        GroupStatus.CORRECTABLE: "train_with_correction",
        GroupStatus.QUARANTINED: "quarantine",
        GroupStatus.REJECTED: "reject",
    }[status]
    rejection_reason: RejectionReason | None = (
        RejectionReason.TOKENIZER_MISMATCH if status == GroupStatus.REJECTED else None
    )
    return GroupDecision(
        group_id=group_id,
        status=status,
        reason=reason,
        rejection_reason=rejection_reason,
        recommended_action=rec_action,
    )


def _make_group(template: dict, *, group_id: str, policy_version: str, lag: int) -> SampleGroup:
    payload = dict(template)
    payload["group_id"] = group_id
    payload["policy_version"] = policy_version
    payload["policy_lag_steps"] = lag
    return SampleGroup.model_validate(payload)


def main() -> None:
    template = json.loads(Path(__file__).with_name("minimal_grpo_group.json").read_text())
    store = InMemoryLiveStore()

    store.submit(
        _make_group(template, group_id="ga", policy_version="v0001", lag=1),
        _decide("ga", GroupStatus.ACCEPTED, "within budget"),
    )
    store.submit(
        _make_group(template, group_id="gc", policy_version="v0001", lag=4),
        _decide("gc", GroupStatus.CORRECTABLE, "high clipped fraction"),
    )
    store.submit(
        _make_group(template, group_id="gold", policy_version="v0000", lag=20),
        _decide("gold", GroupStatus.ACCEPTED, "older policy"),
    )
    store.submit(
        _make_group(template, group_id="gq", policy_version="v0001", lag=2),
        _decide("gq", GroupStatus.QUARANTINED, "low ess"),
    )

    client = TrainerClient(store)

    fresh = client.fetch(
        policy_version="v0001",
        max_groups=10,
        max_staleness=8,
        precision_class="fp32",
        replay_tier=ReplayTier.FRESH,
    )
    print("FRESH batch metadata:")
    print(json.dumps(fresh.metadata(), indent=2))
    print(f"size={fresh.size}, replay_tier={fresh.replay_tier.value}")

    replay = client.fetch(
        policy_version="v0001",
        max_groups=10,
        max_staleness=8,
        replay_tier=ReplayTier.REPLAY,
    )
    print("REPLAY batch metadata:")
    print(json.dumps(replay.metadata(), indent=2))
    print(f"size={replay.size}, replay_tier={replay.replay_tier.value}")


if __name__ == "__main__":
    main()
