# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 5 -- Remote/spot worker hardening. **Complete.** Phase 6 next.

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
- Phase 4.2: Trainer feedback ingestion (FeedbackAggregator).
- Phase 5.1: Heartbeat / lease timeout / audit trail.
- Phase 5.2: k-of-n reload quorum. New pool_reload.PoolReloadTracker
  records per-worker ReloadAcks, exposes ack_count, has_quorum(k, n),
  pool_active_version(k, n), and is_worker_aligned(worker_id, version).
  QuorumViolation raised when two workers ack the same version with
  different checkpoint digests. New docs/k_of_n_reload_quorum.md
  explains pool-level vs. per-group: pool advances on k acks; per-group
  pinning is unchanged and still gated by validate_group_against_lease.
  17 new tests including the load-bearing acceptance test that proves
  per-group pinning is unchanged when the pool has advanced past the
  group's version.

## Next tasks
- Phase 6: Launch demos.
  - Endpoint identity gap dashboard
  - Controlled dense mismatch dashboard
  - Local marketplace simulation with toxic workers and OPBC decisions
  - MoE router mismatch demo

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.
- No live model has been run yet. The endpoint probe and labs accept
  fixture inputs; running them against the spot instance / free-tier
  APIs is a Phase 6 task.

## Test status
- Last run: 2026-05-10, 181 passed, 0 failed (pytest -q)
- ruff check .: clean
