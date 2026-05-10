# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 4 -- OPBC v0. Task 4.1 complete; 4.2 next.

## Completed
- Phase 0 scaffold and docs/configs/examples
- Phase 1.1: PolicyManifest expansion + WorkerManifest. F401 cleanup.
- Phase 1.2: RolloutJob + RolloutLease.
- Phase 1.3: Typed RejectionReason + validators.py.
- Phase 2.1: Endpoint identity gap probe.
- Phase 2.2: Controlled dense mismatch lab (JSON/HTML).
- Phase 2.3: Small MoE router mismatch lab.
- Phase 3.1: LiveStore lifecycle.
- Phase 3.2: Trainer client API.
- Phase 3.3: Worker SDK + LocalWorkerBroker (3-worker concurrent demo).
- Phase 4.1: OPBC decision reasons. New typed DecisionReason enum
  (within_budget, high_clipped_fraction, low_ess, stale_policy_lag,
  replay_tier_lag, replay_tier_ess, veto_threshold_exceeded). GroupDecision
  carries `decision_reasons: list[DecisionReason]` and a fifth
  recommended_action `replay`. decide_group() now emits a graduated
  taxonomy: veto -> quarantine -> train_with_correction -> replay -> train,
  with reasons accumulated where multiple thresholds fire (e.g. low_ess +
  stale_policy_lag together). New BudgetPolicy fields replay_lag_threshold
  (default 4) and replay_ess_threshold (default 0.60). Validators continue
  to handle contract-violation rejections (mixed_policy_version,
  missing_logprobs, etc.) — those still surface as recommended_action
  "reject" with rejection_reason. 17 new tests cover all six acceptance
  scenarios plus replay paths.

## Next tasks
- Phase 4.2: Trainer feedback ingestion (external trainers report ESS,
  clipped fraction, accepted/rejected, and current-policy logprob stats
  after consuming a group; updates per-engine/per-worker quality stats
  without changing trainer logic).
- Phase 5.1: Heartbeat and lease timeout (already partially in Phase 3.3
  via reap_expired; needs audit-trail and timed-out-lease return).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only; opbc.py was rewritten in this PR but
  retains the pre-existing line-length style).

## Test status
- Last run: 2026-05-10, 136 passed, 0 failed (pytest -q)
- ruff check .: clean
