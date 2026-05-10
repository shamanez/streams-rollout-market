# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 3 -- Local market loop. Task 3.1 complete; 3.2 next.

## Completed
- Initial scaffold: contracts, opbc, mismatch_metrics, registry, livestore,
  dispatcher, verifier, worker
- Docs: why_rollout_market, mismatch_observatory, opbc_metrics, evidence_map,
  agent_task_packets, evaluation_plan, protocol, problem_design_survey
- Experiment configs: endpoint probe, controlled dense, MoE router
- Examples: local_worker_demo.py, minimal_grpo_group.json (demo updated to
  use new LiveStore API)
- Phase 1.1: Expanded PolicyManifest; added WorkerManifest. Cleared two
  pre-existing F401 unused-import lint errors.
- Phase 1.2: Added RolloutJob and RolloutLease (idempotency + expiry).
- Phase 1.3: Added typed RejectionReason enum and validators.py module.
- Phase 2.1: Endpoint identity gap probe (observatory + CLI, transport seam).
- Phase 2.2: Controlled dense mismatch lab (Pydantic schema + JSON/HTML + CLI).
- Phase 2.3: Small MoE router mismatch lab (RouterTrace + RouterMismatchReport).
- Phase 3.1: LiveStore lifecycle. Replaced ad-hoc push_group/quarantine_group
  with submit(group, decision) + transition(group_id, new_state, ...) over
  the GroupStatus state machine. Idempotency: duplicate group_id raises
  DuplicateGroupError; decision.group_id must match group.group_id; once
  REJECTED, no further transitions allowed; rejection_reason invariant
  mirrors GroupDecision. get_batch defaults to ACCEPTED+CORRECTABLE in
  insertion order, with include_quarantined=True opt-in for replay-tier
  consumers; REJECTED is never served. count_by_state, has, get,
  StoredGroup.served_count for observability. Updated
  examples/local_worker_demo.py to the new API.

## Next tasks
- Phase 3.2: Trainer client API (consumer pulls valid groups by
  policy_version, max_staleness, precision_class, replay_tier; example fake
  trainer prints batch metadata only).
- Phase 3.3: Worker SDK (registration, heartbeats, lease fetch, group submit,
  failure reporting; three fake workers run concurrently).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 96 passed, 0 failed (pytest -q)
- ruff check .: clean
