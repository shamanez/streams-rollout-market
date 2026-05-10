# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 5 -- Remote/spot worker hardening. Task 5.1 complete; 5.2 next.

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
- Phase 5.1: Heartbeat and lease timeout audit trail. WorkerHealth enum
  (ACTIVE / STALE / DEAD / UNKNOWN) derived from last_seen_at vs.
  heartbeat_ttl_s. New abandon_stale_workers(now) marks dead workers and
  flips their ISSUED leases to ABANDONED (distinct from EXPIRED, which is
  lease-TTL only and does not imply worker death). Abandoned/expired
  leases free their job back to the queue so a fresh worker can pick it
  up. Append-only audit log records worker_registered, worker_revived,
  worker_marked_dead, job_enqueued, lease_issued, lease_submitted,
  lease_failed, lease_expired, lease_abandoned. Dead workers cannot
  acquire new leases until they heartbeat again (revival). 15 new tests
  covering the full lifecycle, expired-vs-abandoned distinction, and
  audit-log content.

## Next tasks
- Phase 5.2: k-of-n reload quorum design doc (per-group policy pinning
  remains strict; per-pool sync may be quorum-based). Design + contract,
  no live coordination required.
- Phase 6: Launch demos (endpoint identity gap dashboard, controlled
  dense dashboard, local marketplace simulation, MoE router demo).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 167 passed, 0 failed (pytest -q)
- ruff check .: clean
