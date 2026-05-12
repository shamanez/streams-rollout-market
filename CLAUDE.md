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

## Architecture (one sentence)

Hermes Agent generates multi-turn trajectories against vLLM-hosted
Qwen3 models; an HTTP proxy (`scripts/live/logprob_capture_proxy.py`)
captures per-token logprobs + token IDs into a sidecar JSONL during
generation; we then teacher-force the **exact same** captured
`(prompt_token_ids, response_token_ids)` through FSDP and Megatron-LM
in bf16 to simulate a trainer's forward pass; the dashboard shows the
resulting per-token logprob shift per `{model, trainer}` tile as
**ESS**. Router-flip-rate (top-k expert-id disagreement) is a TBD
placeholder pending a vLLM upgrade that exposes `routed_experts` on
the OpenAI HTTP endpoint (see `docs/future_research.md`).

## End-to-end reproduction (laptop ↔ spot)

This section is the **only** thing an agent or operator needs to
reproduce everything. Two-host setup: the laptop holds the repo and
runs the merge / pair / render steps; the spot host
(`my-vllm-spot-instance`) holds vLLM, Hermes-Agent, the HF cache,
and the FSDP/Megatron trainer-side scripts.

### Laptop one-time setup

```bash
git clone https://github.com/shamanez/streams-rollout-market.git
cd streams-rollout-market
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check .                   # must be green
```

`~/.ssh/config` must contain a `my-vllm-spot-instance` Host stanza
that resolves to the AWS spot's current IP (the IP changes on
stop/start — never hardcode it).

### Spot one-time setup (per fresh provisioning)

The spot is `g6e.12xlarge` (4× L40S 48 GB, 384 GB RAM, Ubuntu 24.04,
Python 3.12 in `~/rmenv`). Three install pieces:

1. **Hermes-Agent CLI** at `~/.local/bin/hermes` → installs into
   `~/.hermes/hermes-agent/`. Verbatim steps:
   `scripts/live/HERMES_INSTALL.md`. Verified end-to-end working
   against vLLM at 131 K context on 2026-05-12.
2. **vLLM venv** at `~/rmenv/` (Python 3.12, torch 2.11+cu130, vllm
   0.20.2). Re-create with:
   ```bash
   ssh my-vllm-spot-instance 'python3.12 -m venv ~/rmenv && source ~/rmenv/bin/activate && pip install vllm==0.20.2 transformers requests'
   ```
3. **HF model cache** at `~/hf-cache/hub/models--Qwen--<repo>/`
   (canonical HF layout — check this path, NOT `~/hf-cache/<repo>/`).
   Models needed:
    - `Qwen/Qwen3-32B` (Dense)
    - `Qwen/Qwen3-30B-A3B` (MoE)
    - optional FP8 siblings for precision-shift tiles
   Each is ~60 GB. To pre-fetch:
   ```bash
   ssh my-vllm-spot-instance 'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && hf download Qwen/Qwen3-32B && hf download Qwen/Qwen3-30B-A3B'
   ```

The repo's trainer-side scripts live at the spot's home dir (not in
the repo bind-mount) for historical reasons. Push the current revision
once after each `git pull` on the laptop:
```bash
rsync -av scripts/live/ my-vllm-spot-instance:~/streams-rollout-market/scripts/live/
scp scripts/live/run_*.py scripts/live/megatron_*.sh scripts/live/serve_for_capture.sh \
    my-vllm-spot-instance:~/
```

### One full pipeline pass (single bf16 trajectory per model)

This is the load-bearing 8-step pipeline. Each step is a single
command; the whole pass takes ~10 minutes wall-clock per model when
the spot is warm.

```bash
# === 1. (spot) Start vLLM with logprob + token-id capture flags ===
# Dense:
ssh my-vllm-spot-instance 'cd ~ && nohup bash -c "
    source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache &&
    exec vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --dtype auto
        --max-model-len 40960 --gpu-memory-utilization 0.85
        --enable-auto-tool-choice --tool-call-parser hermes
        --host 0.0.0.0 --port 8000
" > ~/vllm-logs/dense_bf16_serve.log 2>&1 &'

# MoE: replace the model line with
#   exec vllm serve Qwen/Qwen3-30B-A3B --tensor-parallel-size 4
#       --enable-expert-parallel --enable-return-routed-experts
#       --no-async-scheduling ...
# (The expert-parallel + no-async-scheduling flags are needed if you
# want routed_experts in CompletionOutput on the native API — but the
# OpenAI HTTP endpoint does NOT serialize them in vLLM 0.20.x.)

# Wait until /v1/models returns 200:
ssh my-vllm-spot-instance 'until curl -sS -o /dev/null -w "%{http_code}\n" \
    http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; do sleep 15; done'

# === 2. (spot) Start the logprob capture proxy on port 8001 ===
ssh my-vllm-spot-instance 'cd ~/streams-rollout-market && source ~/rmenv/bin/activate &&
    nohup python scripts/live/logprob_capture_proxy.py \
        --upstream http://localhost:8000 --port 8001 \
        --sidecar /tmp/hermes_probes.jsonl --host 127.0.0.1 \
        > ~/vllm-logs/proxy.log 2>&1 &'

# === 3. (spot) Drive ONE Hermes-suite task via the non-streaming
#       minimal_agent_driver. (hermes-agent CLI uses streaming, which
#       the proxy can't parse — use the in-repo driver.) ===
ssh my-vllm-spot-instance 'cd ~/streams-rollout-market && source ~/rmenv/bin/activate &&
    python scripts/live/minimal_agent_driver.py \
        --base-url http://localhost:8001 --model Qwen/Qwen3-32B \
        --tasks scripts/live/agent_tasks_hermes.json \
        --task-id no-op-trivia \
        --out /tmp/minimal_sharegpt.jsonl \
        --max-steps 5 --max-tokens 512 --timeout 180'

# === 4. (laptop) Pull captures + merge into AgentTrajectory JSON ===
rsync -a my-vllm-spot-instance:/tmp/minimal_sharegpt.jsonl /tmp/
rsync -a my-vllm-spot-instance:/tmp/hermes_probes.jsonl /tmp/
python3 -c "
import json
tasks = json.load(open('scripts/live/agent_tasks_hermes.json'))
json.dump([t for t in tasks if t['task_id']=='no-op-trivia'],
          open('/tmp/agent_tasks_one.json','w'))
"
mkdir -p runs/live/agent/hermes/qwen3-32b/bf16
python scripts/live/merge_probes_into_trajectories.py \
    --sharegpt /tmp/minimal_sharegpt.jsonl --probes /tmp/hermes_probes.jsonl \
    --tasks /tmp/agent_tasks_one.json --model-id Qwen/Qwen3-32B \
    --engine-label hermes-qwen3-32b-bf16 \
    --engine-fingerprint sha256:hermes-qwen3-32b-bf16-tp4-l40s \
    --out-dir runs/live/agent/hermes/qwen3-32b/bf16/

# === 5. (laptop) Build the trainer-input /tmp/rollouts.json ===
python scripts/live/build_hermes_dense_input.py \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16 \
    --precision-class bf16 \
    --out-rollouts /tmp/rollouts_hermes_dense_bf16.json \
    --out-index    /tmp/hermes_dense_index_bf16.json

# === 6. (spot) Kill vLLM (frees GPUs), push rollouts.json, run FSDP ===
ssh my-vllm-spot-instance 'pkill -9 -f "vllm serve" 2>&1; sleep 5
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$")
    [ -n "$pids" ] && kill -9 $pids; sleep 5'
scp /tmp/rollouts_hermes_dense_bf16.json my-vllm-spot-instance:/tmp/rollouts.json
ssh my-vllm-spot-instance 'cd ~ && source rmenv/bin/activate &&
    export HF_HOME=/home/ubuntu/hf-cache && export OMP_NUM_THREADS=1 &&
    torchrun --nproc-per-node=4 --rdzv-endpoint=127.0.0.1:29510 \
        ~/run_fsdp_reference.py'                          # ~3 min total

# Megatron Dense (docker, ~5 min):
ssh my-vllm-spot-instance 'python3 -c "
import json
d = json.load(open(\"/tmp/rollouts.json\"))[0]
single = {\"model\": d[\"model\"], \"prompt_token_ids\": d[\"prompt_token_ids\"],
          \"response_token_ids\": d[\"response_token_ids\"],
          \"prompt_text\": d.get(\"task_text\", \"\"), \"seed\": 1234}
json.dump(single, open(\"/tmp/rollout.json\", \"w\"))
"
bash ~/megatron_reference_launch.sh'

# === 7. (laptop) Pull trainer JSONs, pair each (rollout, trainer) cell ===
rsync -a my-vllm-spot-instance:/tmp/trainers.json /tmp/trainers_hermes_dense_fsdp_bf16.json
rsync -a my-vllm-spot-instance:/tmp/trainer_megatron-bf16.json /tmp/trainers_hermes_dense_megatron_bf16.json

python scripts/live/pair_hermes_dense_reports.py \
    --index /tmp/hermes_dense_index_bf16.json \
    --trainers /tmp/trainers_hermes_dense_fsdp_bf16.json \
    --trajectory-dir runs/live/agent/hermes/qwen3-32b/bf16 \
    --trainer-engine-label fsdp-bf16 \
    --out-root runs/live/dense/hermes-qwen3-32b-bf16-vs-fsdp-bf16
# (repeat with --trainers .../megatron... and --trainer-engine-label megatron-bf16)

# === 8. (laptop) Re-render the dashboard ===
python scripts/live/publish_dashboards.py
python -m http.server -d docs 8000 &
open http://localhost:8000/
```

The MoE leg of the pipeline is identical with three substitutions:
1. step 1 — `Qwen/Qwen3-30B-A3B` instead of `Qwen/Qwen3-32B`
   (no `--tool-call-parser hermes`, optional `--enable-expert-parallel
   --enable-return-routed-experts --no-async-scheduling`)
2. steps 4 + 5 — use the `runs/live/agent/hermes/qwen3-30b-a3b/`
   tree and the `qwen3-30b-a3b` index suffix.
3. step 6 — `~/run_fsdp_moe_reference.py` ALSO works but writes
   `/tmp/fsdp_router.json` (routed_experts, not logprobs). For the
   logprob-shift dashboard tile we use the SAME `~/run_fsdp_reference.py`
   script (it autoloads any HF causal-LM, including Qwen3MoE).
   Megatron MoE: `~/megatron_moe_reference_launch.sh`.

> **Note** (2026-05-12): Megatron MoE requires the
> `~/megatron_conversion/megatron_ckpt/release/` directory to contain
> the converted Qwen3-30B-A3B torch-dist shards. On a fresh spot
> reprovisioning the dir may be present but empty (HF→Megatron MoE
> conversion not run). Re-run the conversion via
> `~/megatron_conversion/launch_qwen3_30b_a3b.sh` (if present) or
> follow the conversion runbook before the MoE Megatron leg can
> succeed. The dashboard tile for `(Hermes MoE × Megatron)` renders
> as `TBD` until this is done.

The Megatron trainer JSON is a single dict, not a list. After pulling
back to the laptop, wrap it before passing to `pair_hermes_dense_reports.py`:

```bash
python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
if isinstance(d, dict):
    d.setdefault('prompt_idx', 0)
    json.dump([d], open(sys.argv[1], 'w'))
" /tmp/trainers_hermes_dense_megatron_bf16.json
```

### One-command summary (laptop side)

Once the spot is set up and a fresh trajectory is captured, this is
the end-to-end laptop command:

```bash
# Captures live (steps 1-3) are spot-resident; this runs the
# downstream merge → build → push → pair → render on the laptop.
make -f scripts/live/Makefile.demo \
     SPOT=my-vllm-spot-instance \
     MODEL=Qwen/Qwen3-32B \
     TASK=no-op-trivia
```

(A Makefile.demo is on the followup list — for now, run the eight
shell snippets above in order.)

### Why this matters (architecture detail)

vLLM 0.20.x's OpenAI HTTP entrypoint emits `prompt_token_ids` +
per-choice `token_ids` + per-token `logprobs` — that's the entire
capture surface used here. `routed_experts` is exposed only on the
native `LLM.generate()` API path in this vLLM version (`grep -rn
routed_experts vllm/entrypoints/openai/` returns zero hits in 0.20.1
and 0.20.2). That's why the dashboard's router-flip-rate section is a
**TBD placeholder** — it will return once we either (a) upgrade vLLM
to a build that wires routed_experts through the OpenAI shim, or (b)
add a second native-API forward pass that re-feeds the captured token
IDs to harvest routing. See `docs/future_research.md` for the planned
wiring.

### Hermes-Agent CLI vs minimal_agent_driver.py

The repo ships two ways to drive multi-turn trajectories:

| Tool | When to use |
|------|-------------|
| `scripts/live/hermes_agent_runner.py` (wraps `~/.local/bin/hermes`) | Real Hermes-Agent install on the spot. Useful for fidelity tests. Default streaming mode breaks the proxy's `routed_experts` capture — but logprobs land fine. |
| `scripts/live/minimal_agent_driver.py` | **Preferred for the cycle-3 v2 capture pipeline.** Non-streaming OpenAI tool-using driver, identical tool spec to Hermes (calculator, python_eval, read_file, web_search). Every request lands on the proxy as a single `application/json` response so `extract_probes` picks up all four probe fields cleanly. |

Both write the same ShareGPT JSONL format that
`merge_probes_into_trajectories.py` consumes.

### Autonomous loop

```bash
/autonomous-loop              # picks up from PROGRESS.md / STEER.md
touch AGENT_STOP              # halt after current iteration
rm AGENT_STOP                 # resume
bash .claude/scripts/steer.sh "<directive>"   # re-arm STEER.md
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
