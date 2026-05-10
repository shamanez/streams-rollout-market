# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 3 -- Local market loop. **Complete.** Phase 4 next.

## Completed
- Phase 0 scaffold and docs/configs/examples
- Phase 1.1: PolicyManifest expansion + WorkerManifest. Cleared two
  pre-existing F401 unused-import lint errors.
- Phase 1.2: RolloutJob + RolloutLease (idempotency + expiry).
- Phase 1.3: Typed RejectionReason + validators.py.
- Phase 2.1: Endpoint identity gap probe.
- Phase 2.2: Controlled dense mismatch lab (JSON/HTML).
- Phase 2.3: Small MoE router mismatch lab.
- Phase 3.1: LiveStore lifecycle (idempotent submit, transitions, batched
  fetch with QUARANTINED opt-in, REJECTED never served).
- Phase 3.2: Trainer client API (TrainerClient, TrainerBatch, ReplayTier;
  policy_version / max_staleness / precision_class / replay_tier filters).
- Phase 3.3: Worker SDK and broker. New worker_broker.py with thread-safe
  LocalWorkerBroker: registration, heartbeats, capability-matched lease
  issuance, idempotency-key dedup on submit, terminal lease states
  (ISSUED/SUBMITTED/EXPIRED/ABANDONED/FAILED), report_failure that frees
  the job back to the queue, reap_expired for TTL enforcement. Refactored
  worker.py into WorkerSDK (register / heartbeat / fetch_lease /
  submit_group / report_failure). Added examples/three_workers_demo.py:
  three threads drain a 9-job queue, each takes 3 jobs, no races.
  13 new tests including a 15-job concurrent drain.

## Next tasks
- Phase 4.1: OPBC decision reasons (every group gets a typed decision plus
  reasons; tests cover stale, missing-logprob, mixed-version, low-ESS,
  high-clipped-fraction, and clean groups).
- Phase 4.2: Trainer feedback ingestion (external trainers report ESS,
  clipped fraction, accepted/rejected, and current-policy logprob stats
  after consuming a group).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 121 passed, 0 failed (pytest -q)
- ruff check .: clean
