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
