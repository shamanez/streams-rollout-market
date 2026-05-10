# PROGRESS.md -- Agent Handoff State

## Last updated
2026-05-10

## Current phase
Phase 3 -- Local market loop. Tasks 3.1 and 3.2 complete; 3.3 next.

## Completed
- Initial scaffold and Phase 0 docs/configs/examples
- Phase 1.1: Expanded PolicyManifest; added WorkerManifest. Cleared two
  pre-existing F401 unused-import lint errors.
- Phase 1.2: RolloutJob and RolloutLease (idempotency + expiry).
- Phase 1.3: Typed RejectionReason enum + validators.py module.
- Phase 2.1: Endpoint identity gap probe (observatory + CLI).
- Phase 2.2: Controlled dense mismatch lab (Pydantic schema + JSON/HTML + CLI).
- Phase 2.3: Small MoE router mismatch lab.
- Phase 3.1: LiveStore lifecycle. submit (idempotent on group_id), transition
  (terminal REJECTED), get_batch (ACCEPTED+CORRECTABLE default, QUARANTINED
  opt-in), count_by_state, served_count.
- Phase 3.2: Trainer client API. New trainer_client.py with TrainerClient,
  TrainerBatch, ReplayTier. fetch() filters by policy_version (required),
  max_staleness (policy_lag_steps), precision_class, and replay_tier
  (FRESH excludes QUARANTINED, REPLAY admits it; REJECTED never served).
  TrainerBatch.metadata() returns lightweight per-group rows with no token
  payload — safe to log. Module is firewall-clean (no PPO/GRPO/optimizer/
  loss code). Added examples/fake_trainer.py demonstrating FRESH vs REPLAY
  pulls; smoke-tested.

## Next tasks
- Phase 3.3: Worker SDK (registration, heartbeats, lease fetch, group submit,
  failure reporting; three fake workers run concurrently in a local demo).

## Known issues
- src/rollout_market/dispatcher.py and opbc.py fail `ruff format --check`
  (pre-existing formatting only). Not blocking `ruff check .` or pytest.

## Test status
- Last run: 2026-05-10, 108 passed, 0 failed (pytest -q)
- ruff check .: clean
