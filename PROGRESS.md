# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 1 -- Contract-first core. **Complete.** Phase 2 next.

## Completed
- Initial scaffold: contracts, opbc, mismatch_metrics, registry, livestore,
  dispatcher, verifier, worker
- 3 test files: test_contracts, test_opbc, test_mismatch_metrics
- Docs: why_rollout_market, mismatch_observatory, opbc_metrics, evidence_map,
  agent_task_packets, evaluation_plan, protocol, problem_design_survey
- Experiment configs: endpoint probe, controlled dense, MoE router
- Examples: local_worker_demo.py, minimal_grpo_group.json
- Phase 1.1: Expanded PolicyManifest (quantization_class, allowed_engines,
  patch_lineage; non-empty validators on tokenizer_hash, checkpoint_digest,
  policy_version, model_config_hash). Added WorkerManifest. Cleared two
  pre-existing F401 unused-import lint errors.
- Phase 1.2: Added RolloutJob (pinned policy + sampling config + idempotency
  key) and RolloutLease (server-issued ticket with expiry, group_size,
  max_context_tokens, idempotency key, is_expired() helper).
- Phase 1.3: Added typed RejectionReason enum and validators.py module with
  validate_group_against_lease(). Twelve distinct rejection reasons including
  mixed_policy_version, missing_logprobs, missing_token_ids, tokenizer_mismatch,
  precision_mismatch, expired_lease. GroupDecision now requires
  rejection_reason on REJECTED status and forbids it elsewhere.

## Next tasks
- Phase 2.1: Endpoint identity gap probe — CLI calls OpenAI-compatible
  endpoints (Groq/NVIDIA/Cerebras/OpenRouter free tier) and records token /
  logprob contract coverage to runs/<timestamp>/endpoint_contract_report.json.
- Phase 2.2: Controlled dense mismatch lab (depends on 2.1).
- Phase 2.3: MoE router mismatch lab.

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 44 passed, 0 failed (pytest -q)
- ruff check .: clean
