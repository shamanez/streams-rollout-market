# streams-rollout-market

@AGENTS.md

## Project identity

Decentralized rollout marketplace and rollout validity layer for agentic RL.
This is NOT a trainer. See AGENTS.md for the architecture firewall.

## Build commands

```bash
pip install -e ".[dev]"                # install with dev deps
pytest -q                              # run tests
ruff check .                           # lint
ruff format --check .                  # format check
python examples/local_worker_demo.py   # smoke test
```

## Architecture (src/rollout_market/)

| Module | Purpose |
|--------|---------|
| `contracts.py` | Pydantic schemas: PolicyManifest, SampleGroup, Trajectory, WorkerHeartbeat, BudgetReport, GroupDecision, GroupStatus |
| `opbc.py` | Off-policy budget controller: compute_budget_report(), decide_group() |
| `mismatch_metrics.py` | Pure-Python mismatch summary: summarize_logprob_mismatch() |
| `registry.py` | FilePolicyRegistry: publish/latest for policy manifests |
| `livestore.py` | InMemoryLiveStore: push/get_batch/quarantine for sample groups |
| `dispatcher.py` | choose_worker(), make_assignment() for policy-aware dispatch |
| `verifier.py` | canonical_group_hash(), verify_group_hash() for integrity |
| `worker.py` | WorkerRuntime skeleton |

## Development plan

See `streams_rollout_market_git_agent_plan_v2.md` for the Phase 0-6 roadmap.
Track progress in `PROGRESS.md` (maintained by agent between sessions).

## AGENTS.md firewall (MANDATORY)

- Do NOT add PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer, gradient-step, or reward-model-training logic except as mocks under `tests/` or `examples/`.
- Before changing contracts, update docs, fixtures, and tests in the same commit.
- GRPO groups must be one-policy-snapshot by default.

## Code conventions

- Pydantic `BaseModel` for all data contracts
- `dataclasses.dataclass(frozen=True)` for internal config
- Type hints on all public functions
- Docstrings on all public functions
- ruff line-length = 100
- Tests mirror source structure: `test_<module>.py`
- Import from `rollout_market` (not relative) in tests

## Testing requirements

- Every new module must have a corresponding test file
- Every new contract must have at least one fixture and one validation test
- Run `pytest -q && ruff check .` before considering any task complete
- Evidence of test passage is tracked in `.claude/evidence/`

## Session continuity

- Read `PROGRESS.md` at the start of every session
- Update `PROGRESS.md` before ending a session
- Commit all work before stopping
- Check `STEER.md` if it exists (operator redirect)
- Stop immediately if `AGENT_STOP` file exists
