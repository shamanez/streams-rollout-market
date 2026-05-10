# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 1 -- Contract-first core. Task 1.1 complete; 1.2 next.

## Completed
- Initial scaffold: contracts, opbc, mismatch_metrics, registry, livestore,
  dispatcher, verifier, worker
- 3 test files: test_contracts, test_opbc, test_mismatch_metrics
- Docs: why_rollout_market, mismatch_observatory, opbc_metrics, evidence_map,
  agent_task_packets, evaluation_plan, protocol, problem_design_survey
- Experiment configs: endpoint probe, controlled dense, MoE router
- Examples: local_worker_demo.py, minimal_grpo_group.json
- Git initialized with initial commit
- Phase 1.1: Expanded PolicyManifest (added quantization_class, allowed_engines,
  patch_lineage; non-empty validators on tokenizer_hash, checkpoint_digest,
  policy_version, model_config_hash). Added WorkerManifest (static capability
  contract, distinct from WorkerHeartbeat). Added fixtures
  examples/minimal_policy_manifest.json and minimal_worker_manifest.json,
  plus tests/test_manifests.py (12 cases). Cleaned two pre-existing F401
  unused-import lint errors in opbc.py and registry.py.

## Next tasks
- Phase 1.2: Add RolloutJob and RolloutLease
- Phase 1.3: Add RolloutGroup rejection reasons

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 16 passed, 0 failed (pytest -q)
- ruff check .: clean
