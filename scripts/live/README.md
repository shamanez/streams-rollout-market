# Live-run scripts

Drives a real end-to-end Phase 2.1 / 2.2 run: free-tier endpoint probes,
then a real Qwen3-32B dense mismatch comparison between vLLM (rollout) and
HF transformers (reference forward) on the spot instance, then exercises
the full marketplace stack (validators + OPBC + LiveStore + TrainerClient
+ FeedbackAggregator) against the resulting SampleGroup.

The scripts are intentionally minimal — they are operations recipes, not
reusable code. Each one is self-contained.

## Phase 2.1: free-tier endpoint probes (local)

```bash
set -a; source .env; set +a
for prov in groq nvidia cerebras openrouter; do
  python -m rollout_market.cli.endpoint_probe \
    --provider "$prov" \
    --base-url ... --model ... \
    --api-key-env "${prov^^}_API_KEY" \
    --out-root "runs/live/$prov"
done
python -m rollout_market.cli.endpoint_dashboard \
  --reports-glob 'runs/live/*/*/endpoint_contract_report.json' \
  --out-dir runs/live/endpoint_dashboard
```

Findings from the first live round live in `PROGRESS.md` ("Live findings").

## Phase 2.2: Qwen3-32B dense mismatch on the spot instance

Three stages, all keyed off `~/hf-cache` on the spot host (Qwen3-32B is
already cached there).

### 1. vLLM rollout — produces `/tmp/rollout.json`

```bash
scp scripts/live/run_vllm_rollout.py my-vllm-spot-instance:~/
ssh my-vllm-spot-instance \
  'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && \
   python ~/run_vllm_rollout.py'
```

Loads Qwen3-32B in vLLM 0.20+ with TP=4 across 4× L40S, generates one
prompt with `temperature=0.7, top_p=0.95, seed=1234, max_tokens=128,
logprobs=1`, saves `prompt_token_ids`, `response_token_ids`, and
sampled-token logprobs to `/tmp/rollout.json`.

### 2. HF reference forward — produces `/tmp/trainer.json`

```bash
scp scripts/live/run_hf_reference.py my-vllm-spot-instance:~/
ssh my-vllm-spot-instance \
  'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && \
   python ~/run_hf_reference.py'
```

Loads Qwen3-32B in transformers (`bfloat16`, `attn_implementation=sdpa`,
`device_map=auto`), teacher-forces the (prompt + response) tokens,
extracts the trainer-side logprob for each response token from the
preceding-position logits, saves to `/tmp/trainer.json`.

### 3. Pair into a dense_mismatch fixture and run the lab — produces the
report

```bash
scp my-vllm-spot-instance:/tmp/rollout.json /tmp/
scp my-vllm-spot-instance:/tmp/trainer.json /tmp/
python scripts/live/build_dense_input.py
python -m rollout_market.cli.dense_mismatch_lab \
  --input /tmp/dense_mismatch_input.json \
  --out-root runs/live/dense
python -m rollout_market.cli.dense_dashboard \
  --reports-glob 'runs/live/dense/*/dense_mismatch_report.json' \
  --out-dir runs/live/dense_dashboard
```

## Cross-precision: FP8 vs bf16

The same scripts drive a quantized rollout via env vars. Both
`Qwen3-32B-FP8` and `Qwen3-30B-A3B-FP8` are pre-cached on the spot.

```bash
# 1. FP8 vLLM rollout, bf16 HF reference (the realistic case where the
#    worker is quantized to fit on fewer GPUs but the trainer keeps full
#    precision).
ssh my-vllm-spot-instance \
  'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && \
   MODEL=Qwen/Qwen3-32B-FP8 \
   ROLLOUT_LABEL=vllm-fp8 \
   TRAINER_REFERENCE=Qwen/Qwen3-32B \
   python ~/run_vllm_rollout.py'
ssh my-vllm-spot-instance \
  'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && \
   python ~/run_hf_reference.py'

# 2. Pull, build the FP8-flavoured fixture, and run the lab.
scp my-vllm-spot-instance:/tmp/rollout.json /tmp/rollout_fp8.json
scp my-vllm-spot-instance:/tmp/trainer.json /tmp/trainer_fp8.json
python scripts/live/build_dense_input_fp8.py
python -m rollout_market.cli.dense_mismatch_lab \
  --input /tmp/dense_mismatch_input_fp8.json \
  --out-root runs/live/dense
python -m rollout_market.cli.dense_dashboard \
  --reports-glob 'runs/live/dense/*/dense_mismatch_report.json' \
  --out-dir runs/live/dense_dashboard
```

The dashboard then carries one row per (rollout_engine,
trainer_engine) pair — `vllm -> hf-transformers` and
`vllm-fp8 -> hf-transformers` — and the per-pair ESS / clipped /
worst-max-log-ratio columns make the precision drift directly visible.

## sglang as a second rollout engine

A separate venv (`~/sglenv`) keeps sglang's CUDA toolchain isolated
from vLLM. sglang needs nvcc + CUDA libs; the spot host already has a
CUDA-13 toolchain at
`/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13`, so the runner
points `CUDA_HOME` there. sglang's CUDA-graph capture and FlashInfer
JIT both fail without that, so the script also passes
`disable_cuda_graph=True, attention_backend="triton"` (triton has its
own JIT and works without nvcc as long as the linker can find cu13
runtime libs).

```bash
ssh my-vllm-spot-instance \
  'export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
   export PATH=$CUDA_HOME/bin:$PATH
   source ~/sglenv/bin/activate
   export HF_HOME=/home/ubuntu/hf-cache
   MODEL=Qwen/Qwen3-32B-FP8 \
   ROLLOUT_LABEL=sglang-fp8 \
   TRAINER_REFERENCE=Qwen/Qwen3-32B \
   python ~/run_sglang_rollout.py'
ssh my-vllm-spot-instance \
  'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && \
   python ~/run_hf_reference.py'

scp my-vllm-spot-instance:/tmp/rollout.json /tmp/rollout_sglang_fp8.json
scp my-vllm-spot-instance:/tmp/trainer.json /tmp/trainer_sglang_fp8.json
# (also do bf16 in the same shape)
python scripts/live/build_dense_input_sglang.py
python -m rollout_market.cli.dense_mismatch_lab \
  --input /tmp/dense_mismatch_input_sglang_bf16.json \
  --out-root runs/live/dense_sglang_bf16
python -m rollout_market.cli.dense_mismatch_lab \
  --input /tmp/dense_mismatch_input_sglang_fp8.json \
  --out-root runs/live/dense_sglang_fp8
python -m rollout_market.cli.dense_dashboard \
  --reports-glob 'runs/live/dense*/*/dense_mismatch_report.json' \
  --out-dir runs/live/dense_dashboard
```

The dashboard then carries four `(rollout_engine, trainer_engine)`
rows: `vllm`, `vllm-fp8`, `sglang`, `sglang-fp8`, all paired with
`hf-transformers`. The per-pair ESS / mean |Δlogp| / worst max
\|log_ratio\| columns let a reviewer see at a glance which engine ×
precision combinations stay closest to the trainer's logprob view.

## Multi-prompt matrix (8 prompts × 4 engine cells)

The same scripts batch when `PROMPTS_FILE` points at a JSON list of
strings. `scripts/live/prompts.json` carries 8 reasoning + coding +
translation + meta prompts. Single-prompt mode (no `PROMPTS_FILE`)
still works and writes the legacy `/tmp/rollout.json` flat schema.

```bash
# 1. Run each engine cell. Each emits /tmp/rollouts.json + /tmp/trainers.json
#    on the spot, which we pull labelled.
for cell in vllm-bf16 vllm-fp8 sglang-bf16 sglang-fp8; do
    case "$cell" in
        vllm-bf16)   ENV="MODEL=Qwen/Qwen3-32B ROLLOUT_LABEL=vllm-bf16 TRAINER_REFERENCE=Qwen/Qwen3-32B" RUNNER=run_vllm_rollout.py PREP="" ;;
        vllm-fp8)    ENV="MODEL=Qwen/Qwen3-32B-FP8 ROLLOUT_LABEL=vllm-fp8 TRAINER_REFERENCE=Qwen/Qwen3-32B" RUNNER=run_vllm_rollout.py PREP="" ;;
        sglang-bf16) ENV="MODEL=Qwen/Qwen3-32B ROLLOUT_LABEL=sglang-bf16 TRAINER_REFERENCE=Qwen/Qwen3-32B" RUNNER=run_sglang_rollout.py PREP="export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13; export PATH=\$CUDA_HOME/bin:\$PATH;" ;;
        sglang-fp8)  ENV="MODEL=Qwen/Qwen3-32B-FP8 ROLLOUT_LABEL=sglang-fp8 TRAINER_REFERENCE=Qwen/Qwen3-32B" RUNNER=run_sglang_rollout.py PREP="export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13; export PATH=\$CUDA_HOME/bin:\$PATH;" ;;
    esac
    case "$RUNNER" in run_sglang_rollout.py) VENV=sglenv ;; *) VENV=rmenv ;; esac
    ssh my-vllm-spot-instance "$PREP source ~/$VENV/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && cd ~ && PROMPTS_FILE=/home/ubuntu/prompts.json $ENV python $RUNNER"
    ssh my-vllm-spot-instance "source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && cd ~ && python run_hf_reference.py"
    scp my-vllm-spot-instance:/tmp/rollouts.json "/tmp/rollouts.$cell.json"
    scp my-vllm-spot-instance:/tmp/trainers.json "/tmp/trainers.$cell.json"
done

# 2. Build 32 fixtures (4 cells × 8 prompts), one per (engine, prompt) pair.
for cell in vllm-bf16 vllm-fp8 sglang-bf16 sglang-fp8; do
    python scripts/live/build_dense_matrix.py \
        --label "$cell" \
        --rollouts "/tmp/rollouts.$cell.json" \
        --trainers "/tmp/trainers.$cell.json" \
        --out-dir "/tmp/fixtures/$cell"
done

# 3. Run the lab on every fixture, then aggregate.
for f in /tmp/fixtures/*/*.json; do
    python -m rollout_market.cli.dense_mismatch_lab --input "$f" --out-root runs/live/dense
done
python -m rollout_market.cli.dense_dashboard \
    --reports-glob 'runs/live/dense/*/dense_mismatch_report.json' \
    --out-dir runs/live/dense_dashboard
```

The dashboard's per-pair aggregates now reflect means across 8 prompts
× 128 tokens = 1024 tokens per cell, so the per-cell ESS / clipped /
worst max\|log_ratio\| numbers carry actual statistical weight rather
than a single-prompt point estimate.

## MoE router live run (Qwen3-30B-A3B FP8 vs bf16)

The router-mismatch lab needs per-(token, layer, top_k) expert IDs
from both sides. vLLM's chat-completions API does not surface the
router's per-layer decisions, so this run uses HF transformers for
both sides.

`scripts/live/run_moe_router.py` does three stages:

  1. Generate one response from the bf16 checkpoint with sampling
     (temperature 0.7, top_p 0.95, seed 1234) and save the resulting
     prompt + response token IDs.
  2. Teacher-force the (prompt + response) tokens through the bf16
     checkpoint with `output_router_logits=True`, then take top-k of
     the router logits at each MoE layer for every response position.
     This is the *trainer-side* RouterTrace.
  3. Same teacher-force through the FP8 checkpoint
     (Qwen3-30B-A3B-FP8). This is the *rollout-side* RouterTrace.

Spot-host setup notes:
- `pip install -U kernels` is required for HF transformers to load
  the FP8 finegrained MoE checkpoint.
- The HF cache directory must be writable by the runner; on this host
  we ran `sudo chown -R ubuntu:ubuntu ~/hf-cache` once after install.

```bash
scp scripts/live/run_moe_router.py my-vllm-spot-instance:~/
ssh my-vllm-spot-instance \
  'source ~/rmenv/bin/activate && export HF_HOME=/home/ubuntu/hf-cache && \
   python ~/run_moe_router.py'
scp my-vllm-spot-instance:/tmp/router.json /tmp/router.json
python scripts/live/build_router_input.py
python -m rollout_market.cli.router_mismatch_lab \
  --input /tmp/router_mismatch_input.json --out-root runs/live/router
python -m rollout_market.cli.router_dashboard \
  --reports-glob 'runs/live/router/*/router_mismatch_report.json' \
  --out-dir runs/live/router_dashboard
```

## Agent trajectory matrix (multi-step, tool-using)

The trajectory-level lens for the mismatch story. vLLM-or-sglang serves
Qwen3-32B with `--enable-auto-tool-choice`; the Mac orchestrator at
`scripts/live/agent_runner.py` drives a hand-rolled agent loop
(deterministic simulated tools — `web_search`, `calculator`,
`python_eval`, `read_file`) and writes one `AgentTrajectory` JSON per
task. The lab + dashboard from PR #27 do the rest.

```bash
# 1. Start vLLM (or sglang) on the spot with tool calling enabled.
ssh my-vllm-spot-instance \
  'MODEL=Qwen/Qwen3-32B nohup bash ~/serve_qwen_for_agent.sh \
   > /tmp/vllm_serve.log 2>&1 &'
# Wait for "Application startup complete" in the log, then open the
# tunnel.
ssh -f -N -L 8000:localhost:8000 my-vllm-spot-instance

# 2. Run the agent matrix (one engine cell at a time — load each model
#    once, drain all tasks against it).
BASE_URL=http://localhost:8000/v1 \
MODEL=Qwen/Qwen3-32B \
MODEL_ID=Qwen/Qwen3-32B \
ENGINE_LABEL=vllm-bf16 \
ENGINE_FINGERPRINT=sha256:vllm-0.20.1-tp4-bf16-l40s \
OUT_DIR=runs/live/agent/vllm-bf16 \
python scripts/live/agent_runner.py
# Tear down, swap to FP8 / sglang, and repeat.

# 3. Build divergence reports against the bf16 reference.
for engine in vllm-fp8 sglang-bf16 sglang-fp8; do
    for task in $(jq -r '.[].task_id' scripts/live/agent_tasks.json); do
        python -m rollout_market.cli.agent_trajectory_lab \
            --rollout "runs/live/agent/$engine/$task.json" \
            --trainer "runs/live/agent/vllm-bf16/$task.json" \
            --out-root runs/live/agent_diff
    done
done

# 4. Aggregate.
python -m rollout_market.cli.agent_dashboard \
    --reports-glob 'runs/live/agent_diff/*/agent_divergence_report.json' \
    --out-dir runs/live/agent_dashboard
```

Two operational notes:
- `MODEL_ID` is the *logical* checkpoint; `MODEL` is what the API
  expects. Quantized variants (`Qwen/Qwen3-32B-FP8`) and bf16
  reference (`Qwen/Qwen3-32B`) share `MODEL_ID=Qwen/Qwen3-32B` so the
  divergence comparator treats them as the same checkpoint, different
  engine.
- Qwen3 has a "thinking mode" that interleaves chain-of-thought before
  answering. The runner disables it
  (`extra_body={"chat_template_kwargs": {"enable_thinking": False}}`)
  so the 256-token budget is spent on tool calls + concise answers
  instead of CoT.

## Run the dashboard locally

After any of the runs above, you can browse all of the dashboards in
one place:

```bash
python scripts/live/serve_dashboards.py
# Opens http://127.0.0.1:8765/live/index.html in your browser.
# Each dashboard card carries a one-line headline pulled from the
# corresponding JSON, plus a link into the full HTML view.
```

Use `--port 9000` to bind a different port, `--no-browser` to skip
auto-opening, or `--write-only` to regenerate `runs/live/index.html`
without starting a server.

## Marketplace stack validation

```bash
python scripts/live/marketplace_real.py        # bf16 honest + 2 toxic
python scripts/live/marketplace_real_fp8.py    # fp8 vs bf16 manifest + fp8 honest
```

Builds three SampleGroups from the same real rollout (one honest, two
toxic) and pushes them through `validate_group_against_lease`,
`compute_budget_report`, `decide_group`, `InMemoryLiveStore.submit`,
`TrainerClient.fetch`, and `FeedbackAggregator.ingest`. Exits 0 when the
honest group reaches `train` with `within_budget`, the toxic-tokenizer
group is rejected with `tokenizer_mismatch`, and the toxic-no-logprobs
group is rejected with `missing_logprobs`.

## Spot instance setup notes

- `python3.12-venv` and `python3.12-dev` must be installed (`sudo apt
  install`). `python3.12-dev` provides `Python.h`, which Triton's CUDA
  kernel JIT needs.
- vLLM is installed into `~/rmenv` once and reused across runs:
  `python3 -m venv ~/rmenv && source ~/rmenv/bin/activate && pip install
  vllm transformers accelerate`.
- The HF cache is pre-populated at `~/hf-cache`. Both Qwen3-32B and
  Qwen3-30B-A3B (plus FP8 variants) are present.
- Spot instances are reclaimable. Treat `/tmp/*.json` artefacts as
  ephemeral; copy them off the host as soon as a run completes.
