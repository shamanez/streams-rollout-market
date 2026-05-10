# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 1 -- Contract-first core. Tasks 1.1 and 1.2 complete; 1.3 next.

## Completed
- Initial scaffold: contracts, opbc, mismatch_metrics, registry, livestore,
  dispatcher, verifier, worker
- 3 test files: test_contracts, test_opbc, test_mismatch_metrics
- Docs: why_rollout_market, mismatch_observatory, opbc_metrics, evidence_map,
  agent_task_packets, evaluation_plan, protocol, problem_design_survey
- Experiment configs: endpoint probe, controlled dense, MoE router
- Examples: local_worker_demo.py, minimal_grpo_group.json
- Phase 1.1: Expanded PolicyManifest (added quantization_class, allowed_engines,
  patch_lineage; non-empty validators on tokenizer_hash, checkpoint_digest,
  policy_version, model_config_hash). Added WorkerManifest. Fixtures
  examples/minimal_policy_manifest.json and minimal_worker_manifest.json,
  plus tests/test_manifests.py (12 cases). Cleared two pre-existing F401
  unused-import lint errors in opbc.py and registry.py.
- Phase 1.2: Added RolloutJob (pinned policy + sampling config + idempotency
  key, deadline-after-created_at validator) and RolloutLease (server-issued
  ticket with expiry, group_size, max_context_tokens, idempotency key, and
  is_expired() helper). Fixtures examples/minimal_rollout_job.json and
  minimal_rollout_lease.json, plus tests/test_jobs_leases.py (14 cases).

## Next tasks
- Phase 1.3: Add RolloutGroup rejection reasons (typed enum for mixed policy
  version, missing logprobs, missing token IDs, tokenizer mismatch, precision
  mismatch, expired lease)
- Phase 2.1: Endpoint identity gap probe (free-tier providers only)

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 30 passed, 0 failed (pytest -q)
- ruff check .: clean
