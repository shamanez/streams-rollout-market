# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
**Phase 0-6 plan complete.** All four launch demos shipped.

## Completed
- Phase 0 scaffold and docs/configs/examples
- Phase 1.1: PolicyManifest expansion + WorkerManifest. F401 cleanup.
- Phase 1.2: RolloutJob + RolloutLease.
- Phase 1.3: Typed RejectionReason + validators.py.
- Phase 2.1: Endpoint identity gap probe.
- Phase 2.2: Controlled dense mismatch lab.
- Phase 2.3: Small MoE router mismatch lab.
- Phase 3.1: LiveStore lifecycle.
- Phase 3.2: Trainer client API.
- Phase 3.3: Worker SDK + LocalWorkerBroker.
- Phase 4.1: Typed DecisionReason + replay path in OPBC.
- Phase 4.2: Trainer feedback ingestion.
- Phase 5.1: Heartbeat / lease timeout / audit trail.
- Phase 5.2: k-of-n reload quorum.
- Phase 6 (Demo 1): Endpoint identity-gap dashboard.
- Phase 6 (Demo 2): Controlled dense mismatch dashboard.
- Phase 6 (Demo 3): Local marketplace simulation with toxic workers.
- Phase 6 (Demo 4): MoE router mismatch dashboard. New
  observatory/router_dashboard.py aggregates router_mismatch_report.json
  files into a RouterDashboard with per-(rollout_engine, trainer_engine)
  pair aggregates (mean router_flip_rate, mean
  token_expert_disagreement_rate, worst layer flip rate, total tokens,
  layer count) and per-run detail (with layer-flip min/mean/max
  summary). CLI cli/router_dashboard.py consumes a glob and writes
  <out-dir>/router_dashboard.{json,html}. 13 new tests.

## Next tasks
The Phase 0-6 plan is complete. Open follow-ups:
- Run live: produce real endpoint_contract / dense_mismatch /
  router_mismatch reports against the spot instance and free-tier APIs,
  then point the dashboards at them.
- Cleanup: src/rollout_market/dispatcher.py and opbc.py still fail
  `ruff format --check` (pre-existing whitespace only).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only).
- No live model has been run yet. Every test and example uses fixture or
  synthesised data. The contract layer is complete; live data ingestion
  is now a single CLI invocation away on each pipeline.

## Test status
- Last run: 2026-05-10, 232 passed, 0 failed (pytest -q)
- ruff check .: clean
