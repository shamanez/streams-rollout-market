#!/usr/bin/env bash
# Host-side launcher for the Qwen3-32B Megatron-bf16 teacher-force pass.
#
# Mirrors scripts/live/megatron_qwen3_32b_launch.sh (which runs the
# HF→torch-dist conversion); this launcher reuses the same image and
# the same bind-mount layout, but invokes run_megatron_reference.py
# instead of the conversion runner. The script reads /tmp/rollout.json
# inside the container, teacher-forces (prompt + response) through the
# torch-dist checkpoint at TP=4, and writes /tmp/trainer_megatron-bf16.json
# which we copy back to the host.
#
# Usage on the spot host:
#   bash ~/megatron_reference_launch.sh
#
# Default mounts (host → container):
#   $CKPT_ROOT/qwen3-32b/megatron  → /root/Qwen3-32B_torch_dist (ro)
#   <runner script>                → /root/run_megatron_reference.py (ro)
#   <work dir>                     → /tmp_work
#
# Portable: every path is env-driven; $CKPT_ROOT defaults to
# $HOME/checkpoint. Run scripts/live/bootstrap_spot.sh on a fresh spot
# to materialize the directory layout this script expects.
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-$HOME/checkpoint}"
CKPT_DIR="${CKPT_DIR:-$CKPT_ROOT/qwen3-32b/megatron}"
ROLLOUT_FILE_HOST="${ROLLOUT_FILE_HOST:-/tmp/rollout.json}"
TRAINER_OUT_HOST="${TRAINER_OUT_HOST:-/tmp/trainer_megatron-bf16.json}"
RUNNER="${RUNNER:-$(dirname "$(readlink -f "$0")")/run_megatron_reference.py}"
IMAGE="${IMAGE:-slimerl/slime:latest}"
TP="${TENSOR_MODEL_PARALLEL_SIZE:-4}"
HF_CACHE="${HF_CACHE:-$HOME/hf-cache}"

if [[ ! -d "$CKPT_DIR/release" ]]; then
  echo "ERROR: CKPT_DIR=$CKPT_DIR has no release/ subdir " \
       "(was the HF→Megatron conversion completed?)" >&2
  exit 2
fi
if [[ ! -f "$ROLLOUT_FILE_HOST" ]]; then
  echo "ERROR: ROLLOUT_FILE_HOST=$ROLLOUT_FILE_HOST not found " \
       "(run run_vllm_rollout first)" >&2
  exit 2
fi
if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: RUNNER=$RUNNER not found" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
cp "$ROLLOUT_FILE_HOST" "$WORK_DIR/rollout.json"
chmod -R a+r "$WORK_DIR"

echo "[launch] CKPT_DIR=$CKPT_DIR"
echo "[launch] ROLLOUT_FILE_HOST=$ROLLOUT_FILE_HOST"
echo "[launch] WORK_DIR=$WORK_DIR (mounted into container as /tmp_work)"
echo "[launch] TP=$TP"

docker run --rm \
  --gpus all --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$CKPT_DIR":/root/Qwen3-32B_torch_dist:ro \
  -v "$RUNNER":/root/run_megatron_reference.py:ro \
  -v "$WORK_DIR":/tmp_work \
  -v "$HF_CACHE":/root/hf-cache \
  -e MEGATRON_CKPT_DIR=/root/Qwen3-32B_torch_dist \
  -e TENSOR_MODEL_PARALLEL_SIZE="$TP" \
  -e HF_HOME=/root/hf-cache \
  -e HF_TOKENIZER="${HF_TOKENIZER:-Qwen/Qwen3-32B}" \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "$IMAGE" \
  bash -lc "cp /tmp_work/rollout.json /tmp/rollout.json && \
    PYTHONPATH=/root/Megatron-LM/:/root/slime/ \
    torchrun --nproc-per-node $TP /root/run_megatron_reference.py && \
    cp /tmp/trainer_megatron-bf16.json /tmp_work/trainer_megatron-bf16.json && \
    chown \"\$HOST_UID:\$HOST_GID\" /tmp_work/trainer_megatron-bf16.json"

# The host-side destination may be owned by root from a previous run
# (the container writes as root before we add chown). Remove any
# stale file first so the host user can replace it.
rm -f "$TRAINER_OUT_HOST" 2>/dev/null || true
cp "$WORK_DIR/trainer_megatron-bf16.json" "$TRAINER_OUT_HOST"
echo "[launch] wrote $TRAINER_OUT_HOST"
