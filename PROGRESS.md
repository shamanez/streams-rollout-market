# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 2 -- Mismatch Observatory. Task 2.1 complete; 2.2 next.

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
- Phase 2.1: Endpoint identity gap probe. New `observatory/endpoint_probe.py`
  with `EndpointContractReport`, `inspect_chat_completion_response()`,
  `probe_endpoint()` (transport seam for tests), `write_report()`. CLI at
  `cli/endpoint_probe.py` (python -m). Reports to
  runs/<UTC-ts>/endpoint_contract_report.json with provider, model_label,
  token/logprob/top-logprob/seed flags, tokenizer_id, raw_response_hash.
  API key only in Authorization header — excluded from request_hash and
  never persisted. 12 new tests with no live API calls.

## Next tasks
- Phase 2.2: Controlled dense mismatch lab (same checkpoint via reference
  forward + serving backend; compute delta_t, log_ratio, ESS, E[w^2],
  clipped_fraction, max_abs_log_ratio, top_1pct_gradient_mass).
- Phase 2.3: Small MoE router mismatch lab (router_flip_rate,
  token_expert_disagreement_rate).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 56 passed, 0 failed (pytest -q)
- ruff check .: clean
