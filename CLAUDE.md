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
  - **HuggingFace** (HF_TOKEN): model downloads, gated model access (Qwen3, Llama)
- If a model or endpoint requires payment, skip it and document in PROGRESS.md.
- This constraint applies to ALL experiments, tests, and observatory probes.

## Infrastructure constraint: SPOT INSTANCE ONLY (MANDATORY)

- The ONLY AWS resource available is a single spot instance: `ssh my-vllm-spot-instance`
- Instance type: `g6e.12xlarge` — 4x NVIDIA L40S GPUs (24GB each = 96GB total VRAM)
- SSH config is in `~/.ssh/config` (NEVER commit SSH keys, hostnames, or PEM paths to git)
- **DO NOT** create, modify, terminate, or resize any AWS resources
- **DO NOT** run `aws` CLI commands that create/modify infrastructure
- **DO NOT** install heavy dependencies without checking disk space first
- This instance is for running vLLM inference and experiments ONLY
- Spot instances can be terminated by AWS at any time — never store critical data only there
- The hostname is dynamic (changes on stop/start) — always use the SSH alias, not the IP
- Copy experiment results back to local before the instance is reclaimed

## Model selection (decided)

- **Dense model**: `Qwen/Qwen3-32B` — ~64GB bf16, fits on 3 GPUs, available on Groq/Cerebras/OpenRouter
- **MoE model**: `Qwen/Qwen3-30B-A3B` — 30.5B total / 3.3B active, 128 experts (8 active), ~61GB bf16
- Both run in **full bf16 only** — never quantize for Observatory experiments
- Cannot run both simultaneously (~125GB), run sequentially
- Same Qwen3 tokenizer family — tokenizer hash consistency guaranteed
- Use `output_router_logits=True` for MoE expert trace extraction

## Architecture (src/rollout_market/)

| Module | Purpose |
|--------|---------|
| `contracts.py` | Pydantic schemas: PolicyManifest, WorkerManifest, RolloutJob, RolloutLease, SampleGroup, Trajectory, WorkerHeartbeat, BudgetReport, GroupDecision, GroupStatus |
| `opbc.py` | Off-policy budget controller: compute_budget_report(), decide_group() emits typed DecisionReasons (within_budget, low_ess, high_clipped_fraction, replay_tier_*, stale_policy_lag, veto) and routes to train/train_with_correction/replay/quarantine |
| `mismatch_metrics.py` | Pure-Python mismatch summary: summarize_logprob_mismatch() |
| `registry.py` | FilePolicyRegistry: publish/latest for policy manifests |
| `livestore.py` | InMemoryLiveStore: submit/get_batch/transition with idempotent group_id, ACCEPTED+CORRECTABLE served by default, QUARANTINED retained but opt-in |
| `trainer_client.py` | TrainerClient.fetch(policy_version, max_staleness, precision_class, replay_tier) — read-only, firewall-clean |
| `feedback.py` | TrainerFeedback contract + FeedbackAggregator: per-worker / per-engine / per-policy quality stats, idempotent on group_id |
| `pool_reload.py` | PoolReloadTracker: k-of-n weight-sync quorum, is_worker_aligned per-job gate, QuorumViolation on disagreeing digests. Per-group pinning still strict. |
| `dispatcher.py` | choose_worker(), make_assignment() for policy-aware dispatch |
| `verifier.py` | canonical_group_hash(), verify_group_hash() for integrity |
| `validators.py` | validate_group_against_lease() returns typed RejectionReason decisions |
| `observatory/endpoint_probe.py` | probe_endpoint() and EndpointContractReport: contract-coverage probe |
| `cli/endpoint_probe.py` | `python -m rollout_market.cli.endpoint_probe` writes runs/<ts>/endpoint_contract_report.json |
| `observatory/dense_mismatch_lab.py` | DenseMismatchReport contract + JSON/HTML renderer for one paired (rollout, trainer) logprob run |
| `cli/dense_mismatch_lab.py` | `python -m rollout_market.cli.dense_mismatch_lab` writes runs/<ts>/dense_mismatch_report.{json,html} |
| `observatory/router_mismatch_lab.py` | RouterTrace + RouterMismatchReport (router_flip_rate, token_expert_disagreement_rate, per-layer rates) |
| `cli/router_mismatch_lab.py` | `python -m rollout_market.cli.router_mismatch_lab` writes runs/<ts>/router_mismatch_report.{json,html} |
| `worker.py` | WorkerSDK: register / heartbeat / fetch_lease / submit_group / report_failure |
| `worker_broker.py` | LocalWorkerBroker: thread-safe registry, lease issuance, idempotent submission, failure recording, lease reaping (EXPIRED), abandon_stale_workers (ABANDONED), heartbeat-derived WorkerHealth, append-only audit log |

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
