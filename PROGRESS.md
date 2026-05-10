# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 6 -- Launch demos. Marketplace simulation shipped; remaining demos
are dashboard aggregations of the existing labs.

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
- Phase 3.3: Worker SDK + LocalWorkerBroker.
- Phase 4.1: Typed DecisionReason + replay path in OPBC.
- Phase 4.2: Trainer feedback ingestion.
- Phase 5.1: Heartbeat / lease timeout / audit trail.
- Phase 5.2: k-of-n reload quorum.
- Phase 6 (Demo 3 of 4): Local marketplace simulation with toxic workers.
  observatory/marketplace_simulation.py drives the full stack —
  worker_broker + WorkerSDK + validators + opbc + livestore — with seven
  worker profiles (HONEST, NOISY, STALE, TOXIC_TOKENIZER,
  TOXIC_PRECISION, TOXIC_NO_LOGPROBS, TOXIC_VERSION_DRIFT). All five
  recommended_action verdicts are exercised: HONEST -> train, NOISY ->
  train_with_correction (HIGH_CLIPPED_FRACTION), STALE -> quarantine
  (STALE_POLICY_LAG), four toxic profiles -> reject with the matching
  RejectionReason. Output is a self-contained HTML dashboard plus JSON.
  examples/marketplace_simulation_demo.py runs the canonical 9-worker /
  27-job scenario. 13 new tests including profile-by-profile verdict
  assertions and a livestore-state-vs-decision consistency check.

## Next tasks
- Phase 6 (Demo 1): Endpoint identity gap dashboard — aggregate multiple
  endpoint_contract_report.json runs into a single HTML view.
- Phase 6 (Demo 2): Controlled dense mismatch dashboard — aggregate
  multiple dense_mismatch_report runs.
- Phase 6 (Demo 4): MoE router mismatch demo — same shape over
  router_mismatch_report runs.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only).
- No live model has been run yet. The simulation uses synthetic logprobs
  by design; the endpoint probe and dense / MoE labs accept fixture
  inputs and would need a separate "run live, free-tier only" task to
  produce real numbers.

## Test status
- Last run: 2026-05-10, 196 passed, 0 failed (pytest -q)
- ruff check .: clean
