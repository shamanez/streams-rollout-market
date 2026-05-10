"""Phase 4.1 acceptance tests: every group gets a typed decision plus reasons.

Covers the six scenarios named in the plan:
  - clean
  - low-ESS
  - high-clipped-fraction
  - stale (policy_lag)
  - mixed-version (rejected via validators)
  - missing-logprob (rejected via validators)

Plus the new replay path that sits between train and quarantine.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from rollout_market.contracts import (
    BudgetReport,
    DecisionReason,
    GroupStatus,
    PolicyManifest,
    RejectionReason,
    RolloutLease,
    SampleGroup,
)
from rollout_market.opbc import BudgetPolicy, decide_group
from rollout_market.validators import validate_group_against_lease

EXAMPLES = Path(__file__).parents[1] / "examples"


def _report(
    *,
    ess: float = 0.99,
    clipped_fraction: float = 0.0,
    veto_fraction: float = 0.0,
    policy_lag: int = 0,
    max_abs_log_ratio: float = 0.1,
) -> BudgetReport:
    return BudgetReport(
        group_id="g",
        mean_weight=1.0,
        std_weight=0.0,
        second_moment=1.0,
        effective_sample_size=ess,
        clipped_fraction=clipped_fraction,
        veto_fraction=veto_fraction,
        max_abs_log_ratio=max_abs_log_ratio,
        policy_lag_steps=policy_lag,
    )


# --- direct decide_group taxonomy ------------------------------------------------


def test_clean_group_is_train_within_budget():
    decision = decide_group(_report())
    assert decision.recommended_action == "train"
    assert decision.status == GroupStatus.ACCEPTED
    assert decision.decision_reasons == [DecisionReason.WITHIN_BUDGET]


def test_low_ess_quarantines():
    decision = decide_group(_report(ess=0.10))
    assert decision.recommended_action == "quarantine"
    assert decision.status == GroupStatus.QUARANTINED
    assert DecisionReason.LOW_ESS in decision.decision_reasons


def test_high_clipped_fraction_routes_to_correction():
    decision = decide_group(_report(clipped_fraction=0.5))
    assert decision.recommended_action == "train_with_correction"
    assert decision.status == GroupStatus.CORRECTABLE
    assert decision.decision_reasons == [DecisionReason.HIGH_CLIPPED_FRACTION]


def test_stale_policy_lag_quarantines():
    decision = decide_group(_report(policy_lag=20))
    assert decision.recommended_action == "quarantine"
    assert decision.status == GroupStatus.QUARANTINED
    assert DecisionReason.STALE_POLICY_LAG in decision.decision_reasons


def test_replay_tier_lag_routes_to_replay():
    decision = decide_group(_report(policy_lag=6))
    assert decision.recommended_action == "replay"
    assert decision.status == GroupStatus.ACCEPTED
    assert DecisionReason.REPLAY_TIER_LAG in decision.decision_reasons


def test_replay_tier_ess_routes_to_replay():
    decision = decide_group(_report(ess=0.45))
    assert decision.recommended_action == "replay"
    assert decision.status == GroupStatus.ACCEPTED
    assert DecisionReason.REPLAY_TIER_ESS in decision.decision_reasons


def test_replay_can_accumulate_multiple_reasons():
    decision = decide_group(_report(ess=0.45, policy_lag=6))
    assert decision.recommended_action == "replay"
    assert set(decision.decision_reasons) == {
        DecisionReason.REPLAY_TIER_LAG,
        DecisionReason.REPLAY_TIER_ESS,
    }


def test_low_ess_and_stale_attach_both_quarantine_reasons():
    decision = decide_group(_report(ess=0.10, policy_lag=20))
    assert decision.recommended_action == "quarantine"
    assert set(decision.decision_reasons) == {
        DecisionReason.LOW_ESS,
        DecisionReason.STALE_POLICY_LAG,
    }


def test_veto_dominates_other_reasons():
    decision = decide_group(_report(veto_fraction=0.05, ess=0.10, policy_lag=20))
    assert decision.recommended_action == "quarantine"
    assert decision.decision_reasons == [DecisionReason.VETO_THRESHOLD_EXCEEDED]


def test_custom_policy_thresholds_respected():
    strict = BudgetPolicy(replay_ess_threshold=0.99, replay_lag_threshold=0)
    decision = decide_group(_report(ess=0.95, policy_lag=1), policy=strict)
    assert decision.recommended_action == "replay"


def test_decision_reasons_serialize_round_trip():
    decision = decide_group(_report(clipped_fraction=0.5))
    payload = decision.model_dump(mode="json")
    assert "decision_reasons" in payload
    assert payload["decision_reasons"] == ["high_clipped_fraction"]
    assert payload["recommended_action"] == "train_with_correction"


# --- contract violations come from validators, not OPBC --------------------------


def _load_group() -> SampleGroup:
    return SampleGroup.model_validate(
        json.loads((EXAMPLES / "minimal_grpo_group.json").read_text())
    )


def _make_manifest(group: SampleGroup) -> PolicyManifest:
    return PolicyManifest(
        policy_version=group.policy_version,
        checkpoint_digest=group.checkpoint_digest,
        tokenizer_hash=group.tokenizer_hash,
        model_config_hash="cfg-test",
        precision_class=group.precision_class,
        quantization_class=group.quantization_class,
    )


def _make_lease(group: SampleGroup) -> RolloutLease:
    issued = datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)
    return RolloutLease(
        lease_id="lease-1",
        job_id="job-1",
        assignment_id=group.assignment_id,
        worker_id=group.worker_id,
        policy_version=group.policy_version,
        required_precision_class=group.precision_class,
        required_quantization_class=group.quantization_class,
        max_context_tokens=4096,
        group_size=len(group.trajectories),
        idempotency_key="idem-1",
        issued_at=issued,
        expires_at=issued + timedelta(minutes=10),
    )


def test_mixed_version_routes_to_reject_via_validators():
    group = _load_group()
    lease = _make_lease(group)
    manifest = _make_manifest(group).model_copy(update={"policy_version": "v9999"})
    decision = validate_group_against_lease(
        group, lease, manifest, now=datetime(2026, 5, 10, 0, 0, 30, tzinfo=timezone.utc)
    )
    assert decision is not None
    assert decision.recommended_action == "reject"
    assert decision.rejection_reason == RejectionReason.MIXED_POLICY_VERSION
    assert decision.decision_reasons == []


def test_missing_logprobs_routes_to_reject_via_validators():
    group = _load_group()
    new_trajs = [
        t.model_copy(update={"rollout_logprobs": [0.0] * len(t.response_token_ids)})
        for t in group.trajectories
    ]
    group = group.model_copy(update={"trajectories": new_trajs})
    lease = _make_lease(group)
    manifest = _make_manifest(group)
    decision = validate_group_against_lease(
        group, lease, manifest, now=datetime(2026, 5, 10, 0, 0, 30, tzinfo=timezone.utc)
    )
    assert decision is not None
    assert decision.recommended_action == "reject"
    assert decision.rejection_reason == RejectionReason.MISSING_LOGPROBS


def test_clean_group_passes_validators_and_then_train():
    group = _load_group()
    lease = _make_lease(group)
    manifest = _make_manifest(group)
    pre = validate_group_against_lease(
        group, lease, manifest, now=datetime(2026, 5, 10, 0, 0, 30, tzinfo=timezone.utc)
    )
    assert pre is None
    decision = decide_group(_report())
    assert decision.recommended_action == "train"


def test_full_action_taxonomy_covered():
    actions = {
        decide_group(_report()).recommended_action,
        decide_group(_report(ess=0.45)).recommended_action,
        decide_group(_report(clipped_fraction=0.5)).recommended_action,
        decide_group(_report(ess=0.10)).recommended_action,
    }
    assert actions == {"train", "replay", "train_with_correction", "quarantine"}
