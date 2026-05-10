# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 4 -- OPBC v0. **Complete.** Phase 5 next.

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
- Phase 4.1: Typed DecisionReason + replay path in OPBC.
- Phase 4.2: Trainer feedback ingestion. New feedback.py with
  TrainerFeedback (Pydantic contract: group_id + worker_id + engine
  attribution + policy_version + ess_observed + clipped_fraction_observed
  + accepted_by_trainer + trainer_rejection_reason +
  current_policy_logprob_mean/std) and FeedbackAggregator (per-worker,
  per-engine, per-policy QualityStats with running counts and means;
  idempotent on group_id; rejection-reason histogram). Aggregator is
  pure data — no callbacks into trainer/dispatcher/broker, so trainer
  logic is unchanged. examples/fake_trainer.py extended to demo
  ingestion. 16 new tests.

## Next tasks
- Phase 5.1: Heartbeat and lease timeout audit trail (broker already
  reaps expired leases; this task adds explicit abandoned-lease
  reporting and heartbeat-staleness detection).
- Phase 5.2: k-of-n reload quorum design doc (per-group policy pinning
  remains strict; per-pool sync may be quorum-based).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 152 passed, 0 failed (pytest -q)
- ruff check .: clean
