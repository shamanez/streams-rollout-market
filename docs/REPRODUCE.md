# End-to-end reproduction (laptop ↔ spot)

> This is the durable how-to. Read it once when you set up a fresh
> spot or want to re-run the full pipeline. `CLAUDE.md` no longer
> embeds these steps — it just points here.

Two-host setup: the laptop holds the repo and runs the merge / pair
/ render steps; the spot host (`my-vllm-spot-instance`) holds vLLM,
the HF cache, and the FSDP/Megatron trainer-side scripts.

## Laptop one-time setup

```bash
git clone https://github.com/shamanez/streams-rollout-market.git
cd streams-rollout-market
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q && ruff check .                   # must be green
```

`~/.ssh/config` must contain a `my-vllm-spot-instance` Host stanza
that resolves to the AWS spot's current IP (changes on stop/start).

## Spot one-time setup (per fresh provisioning)

The spot is `g6e.12xlarge` (4× L40S 48 GB, 384 GB RAM, Ubuntu 24.04).
**Treated as ephemeral** — every artefact lives in the repo or is
reproducible by `bootstrap_spot.sh` + `restore_megatron_checkpoints.sh`.

Three install pieces on a fresh spot, in this order:

1. **Push the repo:**
   ```bash
   ssh my-vllm-spot-instance 'mkdir -p ~/streams-rollout-market'
   rsync -av --exclude='__pycache__' --exclude='.git' --exclude='runs/' \
         --exclude='docs/' --exclude='.claude/evidence/' \
         ./ my-vllm-spot-instance:~/streams-rollout-market/
   ```
2. **Two vLLM venvs** at `~/rmenv/` (prod) and `~/rmenv-dev/` (patched
   HTTP shim that emits `routed_experts` over the OpenAI HTTP endpoint).
   Python 3.12, torch 2.11+cu130, vllm 0.20.2:
   ```bash
   ssh my-vllm-spot-instance 'python3.12 -m venv ~/rmenv && source ~/rmenv/bin/activate && pip install vllm==0.20.2 transformers requests'
   ssh my-vllm-spot-instance 'python3.12 -m venv ~/rmenv-dev && source ~/rmenv-dev/bin/activate && pip install vllm==0.20.2 transformers requests huggingface_hub && python ~/streams-rollout-market/scripts/live/patch_vllm_routed_experts_http.py "$(python -c "import vllm,os;print(os.path.dirname(vllm.__file__))")"'
   ```
3. **Bootstrap directory layout + HF cache + symlinks** (idempotent):
   ```bash
   ssh my-vllm-spot-instance 'FETCH_HF=1 bash ~/streams-rollout-market/scripts/live/bootstrap_spot.sh'
   ```
   This creates `~/checkpoint/{qwen3-30b-a3b,qwen3-30b-a3b-fp8,qwen3-32b}/{hf,megatron}/`,
   downloads the three Qwen3 models if `FETCH_HF=1`, symlinks
   `<model>/hf` → the canonical snapshot dir, and populates the
   pinned dereferenced HF dir at `~/hf-flat-qwen3-30b-a3b-bf16/`
   (cycle FP8's Megatron MoE launcher uses it via `HF_FLAT_REUSE`).
4. **Restore the Megatron distcp checkpoints** (~117 GB total). Two
   options:
   * From the laptop backup (~30 min):
     ```bash
     bash scripts/live/restore_megatron_checkpoints.sh
     ```
   * Re-convert from the HF cache (~80 min total, GPU-bound):
     ```bash
     ssh my-vllm-spot-instance 'bash ~/streams-rollout-market/scripts/live/megatron_qwen3_32b_launch.sh'
     ssh my-vllm-spot-instance 'bash ~/streams-rollout-market/scripts/live/megatron_qwen3_30b_a3b_launch.sh'
     ```

**Canonical paths after bootstrap:**

| What | Where |
|---|---|
| Repo (trainer + driver scripts) | `~/streams-rollout-market/` |
| HF model cache | `~/hf-cache/hub/models--Qwen--<repo>/snapshots/<sha>/` |
| Per-model convenience root | `~/checkpoint/<model-slug>/{hf,megatron}/` |
| Dereferenced HF for Megatron MoE container | `~/hf-flat-qwen3-30b-a3b-bf16/` |
| vLLM venv (prod) | `~/rmenv/` |
| vLLM venv (HTTP-shim patched) | `~/rmenv-dev/` |

**Env vars every launcher honors:** `CKPT_ROOT` (default
`$HOME/checkpoint`), `HF_HOME` (default `$HOME/hf-cache`).
`HF_FLAT_REUSE` skips the ~60 s `cp -rL` step on the Megatron MoE
launcher.

## One full pipeline pass (single trajectory per precision)

Eight-step pipeline. The numbers below are the cycle-FP8 timings on
a warm spot.

### Cycle FP8 — MoE bf16 + fp8 rollouts × {FSDP, Megatron} trainers

```bash
# === 1. (spot) Serve the bf16 MoE with the patched dev wheel ===
ssh my-vllm-spot-instance 'cd ~ && tmux new-session -d -s vllm "
  bash ~/streams-rollout-market/scripts/live/vllm_serve_moe_dev.sh
"'
# Wait for ready:
ssh my-vllm-spot-instance 'until curl -sS -o /dev/null -w "%{http_code}\n" \
    http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q 200; do sleep 15; done'

# === 2. (spot) Start the logprob capture proxy on port 8001 ===
ssh my-vllm-spot-instance 'cd ~/streams-rollout-market && source ~/rmenv-dev/bin/activate &&
    setsid nohup python scripts/live/logprob_capture_proxy.py \
        --upstream http://localhost:8000 --port 8001 \
        --sidecar /tmp/hermes_probes.jsonl --host 127.0.0.1 \
        > ~/vllm-logs/proxy.log 2>&1 < /dev/null & disown'

# === 3. (spot) Drive the 6-task batch through the proxy ===
# TASKS_HERMES=(math-tokens-per-gpu-day code-fix-factorial mixed-research-and-math plan-3step-experiment)
# TASKS_CWC=(cwc-repo-quality-loop cwc-repo-which-hook-commits)
# Loop calls minimal_agent_driver.py per task; see PROGRESS.md "Cycle FP8" or
# iter03/iter04 evidence for the exact loop.

# === 4. (laptop) Pull + merge into AgentTrajectory JSON ===
mkdir -p runs/live/agent/hermes/qwen3-30b-a3b/bf16
python scripts/live/merge_probes_into_trajectories.py \
    --sharegpt /tmp/minimal_sharegpt_bf16.jsonl \
    --probes /tmp/hermes_probes.jsonl \
    --tasks /tmp/agent_tasks_fp8_batch.json \
    --model-id Qwen/Qwen3-30B-A3B \
    --engine-label hermes-qwen3-30b-a3b-bf16 \
    --engine-fingerprint sha256:hermes-qwen3-30b-a3b-bf16-tp4-l40s \
    --out-dir runs/live/agent/hermes/qwen3-30b-a3b/bf16/

# === 5. (laptop) Build the trainer-input rollouts.json ===
python scripts/live/build_hermes_dense_input.py \
    --trajectory-dir runs/live/agent/hermes/qwen3-30b-a3b/bf16/ \
    --precision-class bf16 \
    --out-rollouts /tmp/rollouts_bf16.json \
    --out-index    /tmp/hermes_moe_index_bf16.json
rsync -a /tmp/rollouts_bf16.json my-vllm-spot-instance:/tmp/

# === 6. (spot) Kill vLLM, run FSDP, run Megatron ===
ssh my-vllm-spot-instance 'pkill -9 -f "vllm serve" 2>&1; sleep 5
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -v "^$")
    [ -n "$pids" ] && kill -9 $pids; sleep 5'
# FSDP MoE:
ssh my-vllm-spot-instance 'cd ~ && source ~/rmenv-dev/bin/activate &&
    export HF_HOME=$HOME/hf-cache && export OMP_NUM_THREADS=1 &&
    cp /tmp/rollouts_bf16.json /tmp/rollouts.json &&
    MODEL=Qwen/Qwen3-30B-A3B \
    FSDP_ROUTER_TRAINERS_OUT=/tmp/trainers_fsdp_moe_bf16.json \
    torchrun --nproc-per-node=4 --rdzv-endpoint=127.0.0.1:29512 \
        ~/streams-rollout-market/scripts/live/run_fsdp_moe_reference.py'
# Megatron MoE:
ssh my-vllm-spot-instance 'HF_FLAT_REUSE=~/hf-flat-qwen3-30b-a3b-bf16 \
    ROLLOUT_FILE_HOST=/tmp/rollouts_bf16.json \
    OUT_SUFFIX=-bf16_rollouts \
    bash ~/streams-rollout-market/scripts/live/megatron_moe_reference_launch.sh'

# Repeat steps 1-6 with fp8 — swap step 1 to vllm_serve_moe_fp8_dev.sh,
# use --engine-label hermes-qwen3-30b-a3b-fp8 in step 4, --precision-class fp8
# in step 5, suffix _fp8 in step 6.

# === 7. (laptop) Pair each rollout × trainer combo ===
rsync -a my-vllm-spot-instance:/tmp/trainers_fsdp_moe_bf16.json /tmp/
rsync -a my-vllm-spot-instance:/tmp/megatron_router-bf16_rollouts.json /tmp/
# Four pair invocations covering bf16×FSDP, bf16×Megatron, fp8×FSDP, fp8×Megatron.
# See PROGRESS.md "Cycle FP8" iter07 evidence for the exact commands.

# === 8. (laptop) Aggregate + republish ===
python -m rollout_market.cli.router_dashboard \
    --reports-glob 'runs/live/router/**/router_mismatch_report.json' \
    --out-dir runs/live/router_dashboard
python scripts/live/publish_dashboards.py
python -m http.server -d docs 8000 &
open http://localhost:8000/
```

## Hermes-Agent CLI vs `minimal_agent_driver.py`

| Tool | When to use |
|------|-------------|
| `scripts/live/hermes_agent_runner.py` (wraps `~/.local/bin/hermes`) | Real Hermes-Agent install on the spot. Default streaming mode breaks the proxy's `routed_experts` capture but logprobs land fine. |
| `scripts/live/minimal_agent_driver.py` | **Preferred for the cycle-3+ capture pipeline.** Non-streaming OpenAI tool-using driver, same tool spec as Hermes (`calculator`, `python_eval`, `read_file`, `web_search`). Every request lands on the proxy as one `application/json` response so the proxy picks up all four probe fields cleanly. |

Both write the same ShareGPT JSONL that
`merge_probes_into_trajectories.py` consumes.

## Why the routed_experts HTTP shim exists

vLLM 0.20.x's OpenAI HTTP entrypoint emits `prompt_token_ids` +
per-choice `token_ids` + per-token `logprobs` out of the box. It does
**not** emit `routed_experts` on the response — that field only exists
on the native `LLM.generate()` API. Upstream
[a7b801e](https://github.com/vllm-project/vllm/blob/a7b801e26d6b9d96bb49e939c0b6b3acf1d85796/docs/training/routed_experts_replay.md)
adds the HTTP-shim wiring, but it requires CUDA 13 (our spot has
12.0). `scripts/live/patch_vllm_routed_experts_http.py` is a
four-file surgical patch that achieves the same outcome on a
prebuilt nightly wheel.

The patch covers gen-side routing only; the proxy trims the engine's
prefill+decode routing buffer down to `usage.completion_tokens`
entries so the sidecar's `response_routed_experts` is 1:1 with
`response_token_ids` — the invariant `pair_hermes_moe_reports.py`
depends on.

## Autonomous loop

```bash
/autonomous-loop              # picks up from PROGRESS.md / STEER.md
touch AGENT_STOP              # halt after current iteration
rm AGENT_STOP                 # resume
bash .claude/scripts/steer.sh "<directive>"   # re-arm STEER.md
```
