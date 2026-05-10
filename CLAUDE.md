# streams-rollout-market

## Project identity

This repository implements a decentralized rollout marketplace and rollout validity layer for agentic RL. It does **not** implement a trainer.

## Build commands

```bash
pip install -e ".[dev]"                # install with dev deps
pytest -q                              # run tests
ruff check .                           # lint
ruff format --check .                  # format check
python examples/local_worker_demo.py   # smoke test
```

## Phase 1 constraint: FREE TIER ONLY (MANDATORY)

- Do NOT make any paid API calls. Only use free-tier models and endpoints.
- Do NOT sign up for paid plans or enter payment information on any provider.
- Respect rate limits — add delays between API calls, never burst.
- Available free-tier providers (API keys stored in .env, NEVER committed):
  - **NVIDIA NIM** (NVIDIA_API_KEY): free inference for many open models
  - **Groq** (GROQ_API_KEY): free fast inference for Llama, Mixtral, Gemma
  - **Cerebras** (CEREBRAS_API_KEY): free fast inference for Llama models
  - **OpenRouter** (OPENROUTER_API_KEY): free-tier models only (check model pricing)
  - **OpenCode** (OPENCODE_API_KEY): free inference endpoint
- If a model or endpoint requires payment, skip it and document in PROGRESS.md.
- This constraint applies to ALL experiments, tests, and observatory probes.

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

## Architecture firewall (MANDATORY)

Do not add PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer, gradient-step, or reward-model-training logic except as mocks under `tests/` or `examples/`.

The core product is:
- worker registration and leases
- rollout dispatch
- policy versioning and weight-sync metadata
- LiveStore
- token/logprob contracts
- rollout verification
- off-policyness and training-inference mismatch telemetry
- trainer-facing client APIs
- Mismatch Observatory experiments

All accepted rollouts must preserve: token IDs, sampled-token behavior logprobs, tokenizer hash, policy version, checkpoint digest, inference engine fingerprint, precision and quantization class, tool-token masks, group membership.

GRPO groups must be one-policy-snapshot by default. Mixed-version groups are invalid unless an explicit experimental contract is used.

Before changing contracts, update docs, fixtures, and tests in the same commit.

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

## Token conservation

- **Sonnet is often at 100% weekly limit** — avoid Sonnet sub-agents when possible
- Prefer `/codex-review` over Claude evaluator for routine reviews
- Use the `haiku` test-runner agent for quick test checks
- Use `opus` only for implementation (implementer agent)
- Run `/stats` or `/usage` periodically to check current usage limits
- Run `bash .claude/scripts/check-usage.sh` to see session-level metrics
- Batch related file changes to reduce tool call count
- For autonomous runs with budget caps: `claude -p --max-budget-usd 5 "/autonomous-loop"`

## Session continuity

- Read `PROGRESS.md` at the start of every session
- Update `PROGRESS.md` before ending a session
- Commit all work before stopping
- Check `STEER.md` if it exists (operator redirect)
- Stop immediately if `AGENT_STOP` file exists
