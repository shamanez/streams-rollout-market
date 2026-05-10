# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 2 -- Mismatch Observatory. Tasks 2.1 and 2.2 complete; 2.3 next.

## Completed
- Initial scaffold: contracts, opbc, mismatch_metrics, registry, livestore,
  dispatcher, verifier, worker
- 3 test files: test_contracts, test_opbc, test_mismatch_metrics
- Docs: why_rollout_market, mismatch_observatory, opbc_metrics, evidence_map,
  agent_task_packets, evaluation_plan, protocol, problem_design_survey
- Experiment configs: endpoint probe, controlled dense, MoE router
- Examples: local_worker_demo.py, minimal_grpo_group.json
- Phase 1.1: Expanded PolicyManifest; added WorkerManifest. Cleared two
  pre-existing F401 unused-import lint errors.
- Phase 1.2: Added RolloutJob and RolloutLease (idempotency + expiry).
- Phase 1.3: Added typed RejectionReason enum and validators.py module with
  validate_group_against_lease() covering 12 distinct rejection branches.
- Phase 2.1: Endpoint identity gap probe — observatory/endpoint_probe.py +
  cli/endpoint_probe.py. EndpointContractReport, transport seam for tests,
  API key isolated from request_hash, CLI writes
  runs/<ts>/endpoint_contract_report.json.
- Phase 2.2: Controlled dense mismatch lab — observatory/dense_mismatch_lab.py
  + cli/dense_mismatch_lab.py. DenseMismatchReport (Pydantic) pinned to
  schema_version "mismatch_observatory.v0", carries identity + all
  acceptance metrics (delta_logprob_mean/abs, sequence_log_ratio, ESS,
  second_moment, clipped_fraction, veto_fraction, max_abs_log_ratio,
  top_1pct_gradient_mass). render_html() emits a self-contained one-page
  view with HTML escaping; write_dense_mismatch() persists JSON + HTML.
  CLI consumes a JSON fixture (examples/dense_mismatch_input.json) — no
  trainer code, no model run required.

## Next tasks
- Phase 2.3: Small MoE router mismatch lab (router trace contract,
  router_flip_rate, token_expert_disagreement_rate). Design + contract +
  metrics — model run can stay deferred.
- Phase 3.1: LiveStore lifecycle (accepted/correctable/quarantine/reject
  with idempotency).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 65 passed, 0 failed (pytest -q)
- ruff check .: clean
