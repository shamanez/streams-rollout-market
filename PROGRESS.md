# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 6 -- Launch demos. Three of four shipped. MoE router demo
remaining.

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
- Phase 6 (Demo 3): Local marketplace simulation with toxic workers.
- Phase 6 (Demo 1): Endpoint identity-gap dashboard.
- Phase 6 (Demo 2): Controlled dense mismatch dashboard. New
  observatory/dense_dashboard.py aggregates dense_mismatch_report.json
  files into a DenseDashboard with per-run table (chronological by
  created_at) and per-(rollout_engine, trainer_engine) aggregate
  (count, mean ESS, mean clipped fraction, mean sequence_log_ratio,
  mean |delta logp|, worst max|log_ratio|, total tokens). Useful for
  answering 'is this engine pair systematically worse than that one?'
  CLI cli/dense_dashboard.py consumes a glob and writes
  <out-dir>/dense_dashboard.{json,html}. 11 new tests.

## Next tasks
- Phase 6 (Demo 4): MoE router mismatch demo — same shape over
  router_mismatch_report.json runs.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only).
- No live model has been run yet. The dashboards are ready to consume
  real reports; producing them requires a separate "run live" pass on
  the spot instance / free-tier APIs.

## Test status
- Last run: 2026-05-10, 218 passed, 0 failed (pytest -q)
- ruff check .: clean
