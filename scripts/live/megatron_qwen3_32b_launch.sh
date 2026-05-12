#!/usr/bin/env bash
# Host-side launcher for the Qwen3-32B HF→Megatron torch-dist conversion.
#
# Runs on `my-vllm-spot-instance` and starts the slimerl/slime container
# with bind-mounts that make the HF snapshot's relative `../../blobs/...`
# symlinks resolve inside the container (previously broken: staging dir
# held host-absolute symlinks that pointed outside the container's
# filesystem, so tokenizer.json was dangling).
#
# Usage on the spot host (from the rsync'd repo path):
#   bash ~/streams-rollout-market/scripts/live/megatron_qwen3_32b_launch.sh
#
# Portable: every path is env-driven. To target a fresh spot, run
# scripts/live/bootstrap_spot.sh first so $CKPT_ROOT/qwen3-32b/megatron
# exists and the HF cache is materialized.
set -euo pipefail

HF_REPO="${HF_REPO:-$HOME/hf-cache/hub/models--Qwen--Qwen3-32B}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/checkpoint}"
CKPT_DIR="${CKPT_DIR:-$CKPT_ROOT/qwen3-32b/megatron}"
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs}"
RUNNER="${RUNNER:-$(dirname "$(readlink -f "$0")")/megatron_qwen3_32b_runner.sh}"
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
