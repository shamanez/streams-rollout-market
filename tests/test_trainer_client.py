import json
from pathlib import Path

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    RejectionReason,
    SampleGroup,
)
from rollout_market.livestore import InMemoryLiveStore
from rollout_market.trainer_client import ReplayTier, TrainerBatch, TrainerClient

EXAMPLES = Path(__file__).parents[1] / "examples"


def _template() -> dict:
    return json.loads((EXAMPLES / "minimal_grpo_group.json").read_text())


def _group(
    *,
    group_id: str = "g",
    policy_version: str = "v0001",
    lag: int = 0,
    precision: str = "fp32",
    quant: str | None = None,
) -> SampleGroup:
    payload = _template()
    payload["group_id"] = group_id
    payload["policy_version"] = policy_version
    payload["policy_lag_steps"] = lag
    payload["precision_class"] = precision
    payload["quantization_class"] = quant
    return SampleGroup.model_validate(payload)


def _decision(group_id: str, status: GroupStatus = GroupStatus.ACCEPTED) -> GroupDecision:
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
        reason="test",
        rejection_reason=rejection_reason,
        recommended_action=rec_action,
    )


def _store(*records: tuple[SampleGroup, GroupStatus]) -> InMemoryLiveStore:
    store = InMemoryLiveStore()
    for group, status in records:
        store.submit(group, _decision(group.group_id, status))
    return store


def test_fetch_filters_by_policy_version():
    store = _store(
        (_group(group_id="a", policy_version="v1"), GroupStatus.ACCEPTED),
        (_group(group_id="b", policy_version="v2"), GroupStatus.ACCEPTED),
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v2", max_groups=10)
    assert isinstance(batch, TrainerBatch)
    assert [r.group_id for r in batch.groups] == ["b"]
    assert batch.requested_policy_version == "v2"


def test_fetch_filters_by_max_staleness():
    store = _store(
        (_group(group_id="a", lag=2), GroupStatus.ACCEPTED),
        (_group(group_id="b", lag=8), GroupStatus.ACCEPTED),
        (_group(group_id="c", lag=12), GroupStatus.ACCEPTED),
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=10, max_staleness=8)
    assert [r.group_id for r in batch.groups] == ["a", "b"]


def test_fetch_filters_by_precision_class():
    store = _store(
        (_group(group_id="a", precision="bf16"), GroupStatus.ACCEPTED),
        (_group(group_id="b", precision="fp32"), GroupStatus.ACCEPTED),
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=10, precision_class="bf16")
    assert [r.group_id for r in batch.groups] == ["a"]


def test_fetch_replay_tier_fresh_excludes_quarantined():
    store = _store(
        (_group(group_id="a"), GroupStatus.ACCEPTED),
        (_group(group_id="b"), GroupStatus.QUARANTINED),
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=10, replay_tier=ReplayTier.FRESH)
    assert [r.group_id for r in batch.groups] == ["a"]


def test_fetch_replay_tier_replay_includes_quarantined():
    store = _store(
        (_group(group_id="a"), GroupStatus.ACCEPTED),
        (_group(group_id="b"), GroupStatus.QUARANTINED),
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=10, replay_tier=ReplayTier.REPLAY)
    assert {r.group_id for r in batch.groups} == {"a", "b"}
    assert batch.replay_tier == ReplayTier.REPLAY


def test_rejected_groups_never_served():
    store = _store(
        (_group(group_id="a"), GroupStatus.ACCEPTED),
        (_group(group_id="b"), GroupStatus.REJECTED),
    )
    client = TrainerClient(store)
    fresh = client.fetch(policy_version="v0001", max_groups=10)
    replay = client.fetch(policy_version="v0001", max_groups=10, replay_tier=ReplayTier.REPLAY)
    assert "b" not in {r.group_id for r in fresh.groups}
    assert "b" not in {r.group_id for r in replay.groups}


def test_correctable_groups_served_in_fresh():
    store = _store(
        (_group(group_id="a"), GroupStatus.CORRECTABLE),
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=10)
    assert [r.group_id for r in batch.groups] == ["a"]


def test_fetch_combines_filters():
    store = _store(
        (_group(group_id="a", policy_version="v1", lag=2, precision="bf16"), GroupStatus.ACCEPTED),
        (_group(group_id="b", policy_version="v1", lag=2, precision="fp32"), GroupStatus.ACCEPTED),
        (_group(group_id="c", policy_version="v1", lag=20, precision="bf16"), GroupStatus.ACCEPTED),
        (_group(group_id="d", policy_version="v2", lag=2, precision="bf16"), GroupStatus.ACCEPTED),
    )
    client = TrainerClient(store)
    batch = client.fetch(
        policy_version="v1",
        max_groups=10,
        max_staleness=4,
        precision_class="bf16",
    )
    assert [r.group_id for r in batch.groups] == ["a"]


def test_fetch_zero_max_groups_returns_empty():
    store = _store((_group(group_id="a"), GroupStatus.ACCEPTED))
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=0)
    assert batch.groups == []
    assert batch.size == 0


def test_fetch_respects_max_groups_limit():
    store = _store(
        *((_group(group_id=f"g{i}"), GroupStatus.ACCEPTED) for i in range(5))
    )
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=3)
    assert [r.group_id for r in batch.groups] == ["g0", "g1", "g2"]


def test_metadata_excludes_token_payload():
    store = _store((_group(group_id="a"), GroupStatus.ACCEPTED))
    client = TrainerClient(store)
    batch = client.fetch(policy_version="v0001", max_groups=10)
    rows = batch.metadata()
    assert len(rows) == 1
    row = rows[0]
    assert "prompt_token_ids" not in row
    assert "trajectories" not in row
    for required in ("group_id", "policy_version", "policy_lag_steps", "state"):
        assert required in row


def test_fetch_increments_served_count():
    store = _store((_group(group_id="a"), GroupStatus.ACCEPTED))
    client = TrainerClient(store)
    client.fetch(policy_version="v0001", max_groups=10)
    client.fetch(policy_version="v0001", max_groups=10)
    assert store.get("a").served_count == 2
