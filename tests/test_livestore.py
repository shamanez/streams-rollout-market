import json
from pathlib import Path

import pytest

from rollout_market.contracts import (
    GroupDecision,
    GroupStatus,
    RejectionReason,
    SampleGroup,
)
from rollout_market.livestore import (
    DuplicateGroupError,
    InMemoryLiveStore,
    StoredGroup,
    UnknownGroupError,
)

EXAMPLES = Path(__file__).parents[1] / "examples"


def _load_group(group_id: str | None = None) -> SampleGroup:
    payload = json.loads((EXAMPLES / "minimal_grpo_group.json").read_text())
    if group_id is not None:
        payload["group_id"] = group_id
    return SampleGroup.model_validate(payload)


def _accepted_decision(group_id: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.ACCEPTED,
        reason="within budget",
        recommended_action="train",
    )


def _correctable_decision(group_id: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.CORRECTABLE,
        reason="train with correction",
        recommended_action="train_with_correction",
    )


def _quarantine_decision(group_id: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.QUARANTINED,
        reason="ess below limit",
        recommended_action="quarantine",
    )


def _rejected_decision(group_id: str) -> GroupDecision:
    return GroupDecision(
        group_id=group_id,
        status=GroupStatus.REJECTED,
        reason="tokenizer mismatch",
        rejection_reason=RejectionReason.TOKENIZER_MISMATCH,
        recommended_action="reject",
    )


def test_submit_stores_record_with_decision_state():
    store = InMemoryLiveStore()
    group = _load_group("g-001")
    record = store.submit(group, _accepted_decision("g-001"))
    assert isinstance(record, StoredGroup)
    assert record.state == GroupStatus.ACCEPTED
    assert record.group_id == "g-001"
    assert record.served_count == 0
    assert len(store) == 1


def test_duplicate_group_id_rejected():
    store = InMemoryLiveStore()
    group = _load_group("g-002")
    store.submit(group, _accepted_decision("g-002"))
    with pytest.raises(DuplicateGroupError):
        store.submit(group, _accepted_decision("g-002"))


def test_submit_requires_decision_id_to_match_group_id():
    store = InMemoryLiveStore()
    group = _load_group("g-003")
    bad_decision = _accepted_decision("not-the-same-id")
    with pytest.raises(ValueError):
        store.submit(group, bad_decision)


def test_submit_rejected_group_records_rejection_reason():
    store = InMemoryLiveStore()
    group = _load_group("g-rej")
    record = store.submit(group, _rejected_decision("g-rej"))
    assert record.state == GroupStatus.REJECTED
    assert record.rejection_reason == RejectionReason.TOKENIZER_MISMATCH


def test_get_batch_serves_accepted_and_correctable_only_by_default():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    store.submit(_load_group("gc"), _correctable_decision("gc"))
    store.submit(_load_group("gq"), _quarantine_decision("gq"))
    store.submit(_load_group("gr"), _rejected_decision("gr"))
    batch = store.get_batch(10)
    ids = [r.group_id for r in batch]
    assert ids == ["ga", "gc"]


def test_get_batch_can_include_quarantined_on_opt_in():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    store.submit(_load_group("gq"), _quarantine_decision("gq"))
    store.submit(_load_group("gr"), _rejected_decision("gr"))
    batch = store.get_batch(10, include_quarantined=True)
    ids = [r.group_id for r in batch]
    assert ids == ["ga", "gq"]
    assert "gr" not in ids


def test_get_batch_respects_max_groups_and_insertion_order():
    store = InMemoryLiveStore()
    for i in range(5):
        gid = f"g{i}"
        store.submit(_load_group(gid), _accepted_decision(gid))
    batch = store.get_batch(3)
    assert [r.group_id for r in batch] == ["g0", "g1", "g2"]


def test_get_batch_increments_served_count():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    store.get_batch(10)
    store.get_batch(10)
    assert store.get("ga").served_count == 2


def test_get_batch_zero_returns_empty():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    assert store.get_batch(0) == []


def test_transition_changes_state_and_reason():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    record = store.transition("ga", GroupStatus.QUARANTINED, "low ess after recheck")
    assert record.state == GroupStatus.QUARANTINED
    assert record.reason == "low ess after recheck"


def test_transition_to_rejected_requires_reason():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    with pytest.raises(ValueError):
        store.transition("ga", GroupStatus.REJECTED, "x")


def test_transition_rejection_reason_only_for_rejected():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    with pytest.raises(ValueError):
        store.transition(
            "ga",
            GroupStatus.QUARANTINED,
            "x",
            rejection_reason=RejectionReason.TOKENIZER_MISMATCH,
        )


def test_transition_out_of_rejected_is_forbidden():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _rejected_decision("ga"))
    with pytest.raises(ValueError):
        store.transition("ga", GroupStatus.ACCEPTED, "appeal")


def test_transition_unknown_group_raises():
    store = InMemoryLiveStore()
    with pytest.raises(UnknownGroupError):
        store.transition("missing", GroupStatus.ACCEPTED, "x")


def test_count_by_state_reports_all_states():
    store = InMemoryLiveStore()
    store.submit(_load_group("ga"), _accepted_decision("ga"))
    store.submit(_load_group("gc"), _correctable_decision("gc"))
    store.submit(_load_group("gq"), _quarantine_decision("gq"))
    store.submit(_load_group("gr"), _rejected_decision("gr"))
    counts = store.count_by_state()
    assert counts[GroupStatus.ACCEPTED] == 1
    assert counts[GroupStatus.CORRECTABLE] == 1
    assert counts[GroupStatus.QUARANTINED] == 1
    assert counts[GroupStatus.REJECTED] == 1
    assert counts[GroupStatus.PENDING] == 0


def test_quarantined_group_is_retained_after_default_batch():
    store = InMemoryLiveStore()
    store.submit(_load_group("gq"), _quarantine_decision("gq"))
    assert store.get_batch(10) == []
    assert store.has("gq")
    record = store.get("gq")
    assert record.state == GroupStatus.QUARANTINED
