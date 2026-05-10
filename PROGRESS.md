# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 6 -- Launch demos. Two of four shipped (marketplace simulation,
endpoint identity-gap dashboard). Two dashboards remaining.

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
- Phase 6 (Demo 1): Endpoint identity-gap dashboard. New
  observatory/endpoint_dashboard.py aggregates endpoint_contract_report
  .json files (latest-per-(provider, model) wins) into an
  EndpointDashboard with capability totals and per-row coverage scores.
  render_html emits a self-contained HTML5 page with a coverage matrix
  (token_ids / sampled_logprobs / top_logprobs / seed_supported × every
  probed pair). New CLI cli/endpoint_dashboard.py consumes a glob and
  writes <out-dir>/endpoint_dashboard.{json,html}. 10 new tests.

## Next tasks
- Phase 6 (Demo 2): Controlled dense mismatch dashboard — same shape over
  dense_mismatch_report.json runs.
- Phase 6 (Demo 4): MoE router mismatch demo — same shape over
  router_mismatch_report.json runs.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only).
- No live model has been run yet. The endpoint-probe and dashboard
  pipeline is wired end-to-end but every report consumed so far is
  fixture-built.

## Test status
- Last run: 2026-05-10, 206 passed, 0 failed (pytest -q)
- ruff check .: clean
