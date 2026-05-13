# streams-rollout-market

> This file is loaded into context on every turn. It only carries
> **load-bearing constraints** (firewalls, infra rules, build cmds)
> and **pointers**. Long-form how-tos live in
> [`docs/REPRODUCE.md`](docs/REPRODUCE.md); recent state lives in
> [`PROGRESS.md`](PROGRESS.md).

## Project identity

Decentralized rollout marketplace and rollout validity layer for
agentic RL. **Does not implement a trainer.**

## Architecture (one sentence)

Hermes-Agent generates multi-turn trajectories against vLLM-hosted
Qwen3 models; a proxy captures per-token logprobs + routed experts
into a sidecar JSONL; we then teacher-force the *same* token IDs
through FSDP-bf16 and Megatron-bf16 to simulate the trainer's
forward pass; the dashboard's headline matrix is per-cell **ESS**
(logprob mismatch) and **router_flip_rate** (top-1 expert disagreement).

## Build commands

```bash
pip install -e ".[dev]"        # install with dev deps
pytest -q                      # 471 tests
ruff check .                   # lint
```

## Architecture firewall (MANDATORY)

Do not add PPO, GRPO, RLOO, DPO, FSDP, Megatron, optimizer,
gradient-step, or reward-model-training **logic** outside `tests/`
or `examples/`. The `scripts/live/run_fsdp_moe_reference.py`,
`scripts/live/run_megatron_moe_reference.py`, etc. are *driver
scripts* that execute existing trainer-side framework code; they
are allowed under `scripts/live/` because they don't define training
logic, only invoke it. New training logic still has to go in
`tests/` or `examples/`.

All accepted rollouts preserve: token IDs, sampled-token behavior
logprobs, tokenizer hash, policy version, checkpoint digest,
inference engine fingerprint, precision and quantization class,
tool-token masks, group membership.

GRPO groups must be one-policy-snapshot by default. Mixed-version
groups are invalid unless an explicit experimental contract is used.

Before changing contracts, update docs, fixtures, and tests in the
same commit.

## Phase 1 constraint: FREE TIER ONLY (MANDATORY)

- Do NOT make paid API calls. Free-tier endpoints only.
- Do NOT enter payment info on any provider.
- Respect rate limits — add delays, never burst.
- Available free-tier providers (API keys in `.env`, never
  committed): **NVIDIA NIM** (`NVIDIA_API_KEY`), **Groq**
  (`GROQ_API_KEY`), **Cerebras** (`CEREBRAS_API_KEY`),
  **OpenRouter** (`OPENROUTER_API_KEY`), **OpenCode**
  (`OPENCODE_API_KEY`), **HuggingFace** (`HF_TOKEN`).

## Infrastructure constraint: SPOT INSTANCE ONLY (MANDATORY)

- The ONLY AWS resource available is one spot instance:
  `ssh my-vllm-spot-instance` (`g6e.12xlarge`, 4× L40S 48 GB).
- SSH config in `~/.ssh/config`. NEVER commit SSH keys, hostnames,
  or PEM paths.
- **DO NOT** create, modify, terminate, or resize any AWS resources.
- **DO NOT** run `aws` CLI commands that mutate infrastructure.
- Spot can be reclaimed at any time — never store critical data only
  there; copy results back to laptop before sessions end.

## Model selection (decided)

- **Dense**: `Qwen/Qwen3-32B` (~64 GB bf16).
- **MoE bf16**: `Qwen/Qwen3-30B-A3B` (30.5 B total / 3.3 B active,
  128 experts × 8 active per token, 48 layers, ~61 GB bf16).
- **MoE FP8**: `Qwen/Qwen3-30B-A3B-FP8` (~31 GB FP8/e4m3, same MoE
  topology). Used as a rollout-side precision; trainer always bf16.
- All three share the Qwen3 tokenizer family (tokenizer hash is
  invariant across them — making bf16/FP8 rollouts directly
  comparable downstream).
- Cannot run dense + MoE simultaneously (~125 GB VRAM); run
  sequentially.

## Code conventions

- Pydantic `BaseModel` for data contracts; `dataclasses.dataclass(frozen=True)` for internal config.
- Type hints + docstrings on public functions.
- ruff line-length 100.
- Tests mirror source: `test_<module>.py`; import from `rollout_market`
  (not relative) in tests.
- Every new module has a test file; every new contract has a fixture
  and a validation test.
- Run `pytest -q && ruff check .` before considering a task complete.
- Evidence of test passage is tracked in `.claude/evidence/`.

## Token conservation

- Sonnet is often at 100% weekly limit — prefer Opus for
  implementation, Codex for routine review (`/codex-review`), Haiku
  for test-runner.
- Run `bash .claude/scripts/check-usage.sh` to see session metrics.

## Session continuity

- Read [`PROGRESS.md`](PROGRESS.md) at start; update before stopping.
- Check `STEER.md` if it exists (operator redirect).
- Stop immediately if `AGENT_STOP` exists.
- Commit all work before stopping.

## Pointers (durable docs)

| Where to look | What's there |
|---|---|
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | Full laptop ↔ spot setup, 8-step pipeline, why the HTTP shim exists, autonomous-loop controls. |
| [`PROGRESS.md`](PROGRESS.md) | Most recent cycle's state + standing notes. |
| `scripts/live/bootstrap_spot.sh` | Idempotent spot setup (venvs, HF cache, pinned dereffed HF, distcp shards). |
| `src/rollout_market/observatory/` | Lab modules — read these to understand contract shapes (`DenseMismatchReport`, `RouterMismatchReport`, `AgentTrajectory`). |
| `src/rollout_market/cli/` | One CLI per lab/dashboard, named after the module. |
| `.claude/skills/` | `autonomous-loop`, `seed-from-plan`. |
| `.claude/evidence/` | Per-iteration write-ups (`iterNN_<slug>/iterNN.md`). |
