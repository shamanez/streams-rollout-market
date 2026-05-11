#!/usr/bin/env bash
# Host-side launcher for the Qwen3-32B HF→Megatron torch-dist conversion.
#
# Runs on `my-vllm-spot-instance` and starts the slimerl/slime container
# with bind-mounts that make the HF snapshot's relative `../../blobs/...`
# symlinks resolve inside the container (previously broken: staging dir
# held host-absolute symlinks that pointed outside the container's
# filesystem, so tokenizer.json was dangling).
#
# Usage on the spot host:
#   ~/megatron_conversion/launch_qwen3_32b.sh
#
# This script lives under scripts/live/ in the repo; copy it (plus the
# inside-container runner) to the spot before invoking:
#   scp scripts/live/megatron_qwen3_32b_launch.sh \
#       my-vllm-spot-instance:~/megatron_conversion/launch_qwen3_32b.sh
#   scp scripts/live/megatron_qwen3_32b_runner.sh \
#       my-vllm-spot-instance:~/megatron_conversion/runner_qwen3_32b.sh
set -euo pipefail

HF_REPO="${HF_REPO:-$HOME/hf-cache/hub/models--Qwen--Qwen3-32B}"
CKPT_DIR="${CKPT_DIR:-$HOME/megatron_conversion/qwen3_32b_torch_dist}"
LOG_DIR="${LOG_DIR:-$HOME/megatron_conversion/logs}"
RUNNER="${RUNNER:-$HOME/megatron_conversion/runner_qwen3_32b.sh}"
IMAGE="${IMAGE:-slimerl/slime:latest}"

if [[ ! -d "$HF_REPO/snapshots" ]]; then
  echo "ERROR: HF_REPO=$HF_REPO is not a HuggingFace hub repo (no snapshots/)" >&2
  exit 2
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: RUNNER=$RUNNER missing or not executable" >&2
  exit 2
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"

LOG_FILE="$LOG_DIR/qwen3_32b_convert.out"
echo "[launch] HF_REPO=$HF_REPO"
echo "[launch] CKPT_DIR=$CKPT_DIR"
echo "[launch] LOG_FILE=$LOG_FILE"

# Bind-mount the entire repo (not just the snapshot dir) so that the
# `../../blobs/...` relative symlinks inside snapshots/<rev>/ resolve.
exec docker run --rm \
  --gpus all --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_REPO":/root/Qwen3-32B-repo:ro \
  -v "$CKPT_DIR":/root/Qwen3-32B_torch_dist \
  -v "$LOG_DIR":/root/logs \
  -v "$RUNNER":/root/runner_qwen3_32b.sh:ro \
  "$IMAGE" \
  bash -lc "/root/runner_qwen3_32b.sh 2>&1 | tee /root/logs/qwen3_32b_convert.out"
