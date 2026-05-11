# Convert Qwen3-30B-A3B to a Megatron torch-dist checkpoint

STEER `megatron.qwen3_moe_conversion_runbook` — operations recipe.

The dense matrix already pairs **vLLM (rollout)** with **FSDP** as a
trainer-side reference. FSDP shards parameters but all-gathers them on
the forward pass, so its logprobs are bit-identical to HF
transformers — no new evidence about train-vs-inference equivalence.
**Megatron-LM** is the first trainer engine that actually moves the
forward-pass numbers (TP / EP / SP all change the GEMM ordering and
softmax precision).

This runbook gets a Megatron-shaped checkpoint of
`Qwen/Qwen3-30B-A3B` (the MoE model — 30.5 B total, 3.3 B active,
128 experts, top-k 8) onto the spot instance so the existing live
scripts can feed it into `router_mismatch_lab` and `dense_mismatch_lab`
without any code changes.

## Prerequisites

| Item                | Value / source                                  |
| ------------------- | ----------------------------------------------- |
| Instance            | `ssh my-vllm-spot-instance` (`g6e.12xlarge`, 4 × L40S, 96 GB total VRAM) |
| Docker image (pinned)| `slimerl/slime:latest`                          |
| HF model            | `Qwen/Qwen3-30B-A3B` (pre-pulled into `~/hf-cache` on the spot) |
| Disk budget         | **~200 GB peak** (HF snapshot ~60 GB + Megatron torch-dist ~120 GB + scratch) |
| Time budget         | **~60 min** HF download (if cache cold) + **~20 min** torch-dist conversion |
| GPU budget          | 4 × L40S during conversion (TP=4, EP=1)         |

The slimerl/slime image carries a pinned Megatron-LM tree at
`/root/Megatron-LM` plus the matching torch / CUDA toolchain. **Do
not** mix it with the project's `rmenv` venv — Megatron expects its
own CUDA stack.

## Host directory layout

Make a per-run scratch directory on the spot instance:

```bash
ssh my-vllm-spot-instance '
  mkdir -p ~/megatron_conversion/hf_model \
           ~/megatron_conversion/megatron_ckpt \
           ~/megatron_conversion/logs
'
```

The three subdirectories map 1-to-1 to volume mounts inside the
docker container (next section). Keep them on the same EBS volume as
`~/hf-cache` to avoid an extra copy step.

## Step 1. Stage the HF snapshot

The image's working directory expects the HF model at
`/root/Qwen3-30B-A3B/`. The fastest path is to copy (or symlink) from
the existing `~/hf-cache`:

```bash
ssh my-vllm-spot-instance '
  set -e
  if [ ! -f ~/megatron_conversion/hf_model/config.json ]; then
    HF_SRC="$(ls -d ~/hf-cache/hub/models--Qwen--Qwen3-30B-A3B/snapshots/*/ | head -1)"
    cp -r "$HF_SRC"/* ~/megatron_conversion/hf_model/
  fi
'
```

If the cache is cold (no Qwen3-30B-A3B in `~/hf-cache`), seed it first
with `huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir
~/megatron_conversion/hf_model`. Budget ~60 min for the download.

## Step 2. Run the docker container

```bash
ssh my-vllm-spot-instance \
  'docker run --rm --gpus all --ipc=host \
     --shm-size=64g \
     --ulimit memlock=-1 --ulimit stack=67108864 \
     -v ~/megatron_conversion/hf_model:/root/Qwen3-30B-A3B \
     -v ~/megatron_conversion/megatron_ckpt:/root/Qwen3-30B-A3B_torch_dist \
     -v ~/megatron_conversion/logs:/root/logs \
     slimerl/slime:latest \
     bash -lc "
       PYTHONPATH=/root/Megatron-LM/ \
       torchrun --nproc-per-node 4 \
         /root/Megatron-LM/tools/convert_hf_to_torch_dist.py \
         \${MODEL_ARGS[@]} \
         --hf-checkpoint /root/Qwen3-30B-A3B/ \
         --save /root/Qwen3-30B-A3B_torch_dist/ \
         2>&1 | tee /root/logs/convert.log
     "'
```

Flags are deliberate and **must not be skipped**:

- `--gpus all` — the conversion uses all four L40S as TP=4.
- `--ipc=host` — PyTorch DataLoader's shared memory.
- `--shm-size=64g` — without this, NCCL falls back to host RAM and OOMs.
- `--ulimit memlock=-1 --ulimit stack=67108864` — Megatron pins large
  pages for the all-gather buffers.

`${MODEL_ARGS[@]}` is the per-model knob list the slimerl/slime image
publishes for Qwen3 MoE — `--target-tensor-parallel-size 4
--target-expert-model-parallel-size 1 --target-pipeline-parallel-size 1
--swiglu --num-experts 128 --topk 8 --hidden-size 2048
--ffn-hidden-size 768 --num-layers 48 --num-attention-heads 32
--seq-length 2048` and so on. The image's bashrc exports a default
`MODEL_ARGS` for Qwen3 MoE; verify it matches the HF config before
the run (`echo $MODEL_ARGS` inside the container).

## Step 3. Hand off to the live scripts

The existing live scripts (`scripts/live/run_vllm_rollout.py`,
`scripts/live/run_fsdp_reference.py`, and the new
`scripts/live/run_megatron_reference.py` — to follow in a separate
PR) already read these env vars:

| Env var                 | Value                                       |
| ----------------------- | ------------------------------------------- |
| `MODEL`                 | `Qwen/Qwen3-30B-A3B`                        |
| `ROLLOUT_LABEL`         | `vllm-bf16`                                 |
| `TRAINER_REFERENCE`     | `megatron` (a new value the runbook adds)   |
| `MEGATRON_CKPT_DIR`     | `/root/Qwen3-30B-A3B_torch_dist` (in container) or `~/megatron_conversion/megatron_ckpt` (on host) |
| `DEVICE_VENDOR`         | `NVIDIA`                                    |
| `DEVICE_FAMILY`         | `L40S`                                      |
| `DEVICE_CUDA_COMPUTE`   | `8.9`                                       |
| `DEVICE_VRAM_GB`        | `24`                                        |
| `DEVICE_HOST_LABEL`     | `g6e.12xlarge`                              |

Once the trainer reference is live, the existing
`scripts/live/build_dense_input.py --variant megatron-bf16` writes
the dense-mismatch input that the lab CLI consumes (variant added in
STEER `cleanup.consolidate_build_dense_input`).

## Step 4. Cleanup (spot reclaim hygiene)

Spot instances are reclaimable. Keep the run artefacts off the spot
host:

```bash
# Pull the dist-ckpt off the spot before AWS reclaims it.
scp -r my-vllm-spot-instance:~/megatron_conversion/megatron_ckpt \
  ~/megatron_conversion/megatron_ckpt

# On a successful run, the on-spot scratch can go.
ssh my-vllm-spot-instance 'rm -rf ~/megatron_conversion/hf_model ~/megatron_conversion/logs'

# On reclaim, the megatron_ckpt directory needs to be re-uploaded — keep
# the local copy as the canonical artefact.
```

If a Megatron-Core release pushes torch-dist layout changes, the dist
checkpoint is **not** forward-compatible; regenerate with the new
image tag.

## Why this is operationally important

The dense + MoE matrices on `/docs/index.html` carry MEGATRON rows
that today render the renderer-generated placeholder
`TBD — pending HF→Megatron conversion` (STEER
`dashboard.megatron_placeholder`). Running this runbook end-to-end
fills the MEGATRON row with real data, which is the first numeric
evidence in the repo that a forward-pass-moving trainer engine
disagrees with vLLM on the same checkpoint.

That answer determines whether the "engines disagree even at the
forward pass" claim has teeth or is HF-only.

## Cross-references

- `scripts/live/README.md` — runbook in context of the live scripts.
- `docs/future_research.md` — open question #3 (Megatron-LM as the
  fourth trainer-side reference).
- Dashboards: `docs/router_dashboard.html` and `docs/dense_dashboard.html`
  will populate their MEGATRON rows once `TRAINER_REFERENCE=megatron`
  reports land under `runs/live/router/` and `runs/live/dense/`.
- STEER `dashboard.dense_moe_split` and `dashboard.megatron_placeholder`
  define the dashboard surface this run feeds.
