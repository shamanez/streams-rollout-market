# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 2 -- Mismatch Observatory. **Complete.** Phase 3 next.

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
- Phase 1.3: Added typed RejectionReason enum and validators.py module.
- Phase 2.1: Endpoint identity gap probe — observatory/endpoint_probe.py +
  cli/endpoint_probe.py. Transport seam for tests, API key isolated.
- Phase 2.2: Controlled dense mismatch lab — DenseMismatchReport (Pydantic,
  schema_version "mismatch_observatory.v0") + JSON/HTML renderer + CLI.
- Phase 2.3: Small MoE router mismatch lab — RouterTrace contract
  (per-token, per-layer, top-k expert ids) + RouterMismatchReport
  (router_flip_rate, token_expert_disagreement_rate, per-layer flip rates) +
  JSON/HTML renderer + CLI. Schema-validated trace shape, top-k vs top-1
  semantics distinguish ordering from set membership.

## Next tasks
- Phase 3.1: LiveStore lifecycle (accepted/correctable/quarantine/reject
  with idempotency; duplicate group id rejected; quarantined groups
  retained but not served by default).
- Phase 3.2: Trainer client API.
- Phase 3.3: Worker SDK.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 80 passed, 0 failed (pytest -q)
- ruff check .: clean
