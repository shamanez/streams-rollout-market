# streams-rollout-market

## Project identity

This repository implements a decentralized rollout marketplace and rollout validity layer for agentic RL. It does **not** implement a trainer.

## Build commands

```bash
pip install -e ".[dev]"                # install with dev deps
pytest -q                              # run tests (347 passing)
ruff check .                           # lint
ruff format --check .                  # format check
python examples/local_worker_demo.py   # smoke test
```

## How to run

### View the dashboard locally

```bash
# 1. Re-render docs/index.html from the latest runs/live/*.json bundles.
python scripts/live/publish_dashboards.py            # writes to docs/
# 2. Serve docs/ and open the index in a browser.
python -m http.server -d docs 8000 &
open http://localhost:8000/
```

The shipped `docs/index.html` answers one question above the fold:
*does engine + precision + device change inference-time behaviour
enough to matter for crowdsourced MoE rollouts?* Two matrices (dense
Qwen3-32B + MoE Qwen3-30B-A3B), four green/amber tiles each, traffic-
light coloured against per-metric thresholds. Headline numbers come
from `runs/live/{dense,router,agent}_dashboard.{html,json}`.

### Dense matrix (Qwen3-32B) — vLLM rollout vs FSDP/Megatron reference

Each cell is one `live(...)` commit. Env-driven, runs on
`my-vllm-spot-instance`. Outputs land in `runs/live/dense/`.

```bash
# vLLM-bf16 rollout (writes /tmp/rollout.json on the spot)
ssh my-vllm-spot-instance 'cd ~ && VLLM_DTYPE=bfloat16 \
  python scripts/live/run_vllm_rollout.py'

# FSDP-bf16 teacher-force (writes /tmp/trainer.json)
ssh my-vllm-spot-instance 'cd ~ && torchrun --nproc-per-node 4 \
  scripts/live/run_fsdp_reference.py'

# Megatron-bf16 teacher-force inside slimerl/slime container
ssh my-vllm-spot-instance 'bash scripts/live/megatron_reference_launch.sh'

# Pair them into a DenseMismatchReport and render the tile.
python scripts/live/build_dense_input.py --variant vllm-bf16 \
  --out runs/live/dense/vllm-bf16-fsdp-bf16.json
python -m rollout_market.cli.dense_dashboard \
  --reports-glob 'runs/live/dense/*.json' \
  --out-dir runs/live/dense_dashboard
```

Variants supported by `build_dense_input.py --variant`:
`{vllm-bf16, vllm-fp8, sglang-bf16, sglang-fp8, megatron-bf16}`.
Currently shipped tiles: 4 (FSDP × {bf16,fp8} + Megatron × {bf16,fp8}).

### MoE matrix (Qwen3-30B-A3B) — router_flip_rate per (engine × precision)

Mirror of the dense path but with router-trace emission. vLLM-fp8 MoE
runs at TP=2 (TP=4 fails vLLM's FP8 `block_n=128` constraint).

```bash
# vLLM MoE rollout with router traces (writes /tmp/vllm_router.json)
ssh my-vllm-spot-instance 'cd ~ && VLLM_DTYPE=bfloat16 \
  python scripts/live/run_vllm_moe_rollout.py'

# FSDP MoE reference (writes /tmp/fsdp_router.json)
ssh my-vllm-spot-instance 'cd ~ && torchrun --nproc-per-node 4 \
  scripts/live/run_fsdp_moe_reference.py'

# Megatron MoE reference inside slimerl/slime container
ssh my-vllm-spot-instance 'bash scripts/live/megatron_moe_reference_launch.sh'

# Pair into a RouterMismatchReport; --variant picks one of four cells.
python scripts/live/build_router_input_pair.py \
  --rollout-input /tmp/vllm_router.json \
  --trainer-input /tmp/fsdp_router.json \
  --variant vllm-bf16-fsdp-bf16 \
  --out runs/live/router/<variant>.json
python -m rollout_market.cli.router_dashboard \
  --reports-glob 'runs/live/router/*.json' \
  --out-dir runs/live/router_dashboard
```

Variants: `{vllm-bf16, vllm-fp8} × {fsdp-bf16, megatron-bf16}` (4 cells).

### Autonomous loop

```bash
/autonomous-loop              # picks up from PROGRESS.md "Next work"
touch AGENT_STOP              # halt after current iteration
rm AGENT_STOP                 # resume
bash .claude/scripts/steer.sh "<directive>"   # re-arm STEER.md
```

### Free-tier endpoint probes (legacy / appendix)

```bash
set -a; source .env; set +a
python -m rollout_market.cli.endpoint_probe --provider groq \
  --base-url https://api.groq.com/openai/v1 \
  --model llama-3.3-70b-versatile --api-key-env GROQ_API_KEY \
  --out-root runs/live/groq
```

`scripts/live/README.md` has the full per-provider matrix.

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
| `observatory/marketplace_simulation.py` | End-to-end concurrent simulation with honest/noisy/stale/toxic worker profiles; aggregates verdicts to JSON+HTML dashboard |
| `observatory/endpoint_dashboard.py` | Aggregates many endpoint_contract_report.json files into one HTML+JSON dashboard (provider × model coverage matrix) |
| `cli/endpoint_dashboard.py` | `python -m rollout_market.cli.endpoint_dashboard --reports-glob ... --out-dir ...` |
| `observatory/dense_dashboard.py` | Aggregates dense_mismatch_report.json files: per-(rollout_engine, trainer_engine) pair stats + per-run table |
| `cli/dense_dashboard.py` | `python -m rollout_market.cli.dense_dashboard --reports-glob ... --out-dir ...` |
| `observatory/router_dashboard.py` | Aggregates router_mismatch_report.json files: per-pair flip rates + per-run layer-level summary |
| `cli/router_dashboard.py` | `python -m rollout_market.cli.router_dashboard --reports-glob ... --out-dir ...` |
| `observatory/agent_trajectory_lab.py` | AgentStep / AgentTrajectory / ToolCallRecord contracts, compute_trajectory_divergence, side-by-side HTML diff |
| `observatory/agent_dashboard.py` | Aggregates agent_divergence_report.json files: per-(rollout, trainer) first-div-step / tool-jaccard / answer-match aggregates |
| `cli/agent_trajectory_lab.py` | `python -m rollout_market.cli.agent_trajectory_lab --rollout ... --trainer ...` |
| `cli/agent_dashboard.py` | `python -m rollout_market.cli.agent_dashboard --reports-glob ... --out-dir ...` |
| `scripts/live/serve_dashboards.py` | One-command launcher: writes runs/live/index.html linking every dashboard, then serves runs/ over http and opens the index |
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

See `docs/git_plan_v2.md` for the Phase 0-6 roadmap (now complete).
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
