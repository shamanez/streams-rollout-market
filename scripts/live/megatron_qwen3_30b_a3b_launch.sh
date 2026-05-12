#!/usr/bin/env bash
# Host-side launcher for the Qwen3-30B-A3B (MoE) HF→Megatron torch-dist
# conversion.
#
# Mirror of scripts/live/megatron_qwen3_32b_launch.sh — same image,
# same bind-mount pattern, different MODEL_ARGS (loaded inside the
# container by the sourced slime model script). Produces a torch-dist
# checkpoint under $CKPT_ROOT/qwen3-30b-a3b/megatron that the
# downstream Megatron MoE teacher-force pass (megatron_moe_reference_launch.sh)
# consumes.
#
# Usage on the spot host (from the rsync'd repo path):
#   bash ~/streams-rollout-market/scripts/live/megatron_qwen3_30b_a3b_launch.sh
#
# Portable: every path is env-driven. To target a fresh spot, run
# scripts/live/bootstrap_spot.sh first so the HF cache is materialized
# and the destination directory exists.
set -euo pipefail

HF_REPO="${HF_REPO:-$HOME/hf-cache/hub/models--Qwen--Qwen3-30B-A3B}"
CKPT_ROOT="${CKPT_ROOT:-$HOME/checkpoint}"
CKPT_DIR="${CKPT_DIR:-$CKPT_ROOT/qwen3-30b-a3b/megatron}"
LOG_DIR="${LOG_DIR:-$CKPT_ROOT/logs}"
RUNNER="${RUNNER:-$(dirname "$(readlink -f "$0")")/megatron_qwen3_30b_a3b_runner.sh}"
IMAGE="${IMAGE:-slimerl/slime:latest}"

if [[ ! -d "$HF_REPO/snapshots" ]]; then
  echo "ERROR: HF_REPO=$HF_REPO is not a HuggingFace hub repo (no snapshots/)" >&2
  echo "Run: hf download Qwen/Qwen3-30B-A3B (or scripts/live/bootstrap_spot.sh)" >&2
  exit 2
fi
if [[ ! -x "$RUNNER" ]]; then
  echo "ERROR: RUNNER=$RUNNER missing or not executable" >&2
  exit 2
fi

mkdir -p "$CKPT_DIR" "$LOG_DIR"

LOG_FILE="$LOG_DIR/qwen3_30b_a3b_convert.out"
echo "[launch] HF_REPO=$HF_REPO"
echo "[launch] CKPT_DIR=$CKPT_DIR"
echo "[launch] LOG_FILE=$LOG_FILE"

# Bind-mount the entire HF cache repo (not just the snapshot dir) so
# the `../../blobs/...` relative symlinks inside snapshots/<rev>/
# resolve inside the container.
exec docker run --rm \
  --gpus all --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$HF_REPO":/root/Qwen3-30B-A3B-repo:ro \
  -v "$CKPT_DIR":/root/Qwen3-30B-A3B_torch_dist \
  -v "$LOG_DIR":/root/logs \
  -v "$RUNNER":/root/runner_qwen3_30b_a3b.sh:ro \
  "$IMAGE" \
  bash -lc "ln -sf /root/Qwen3-30B-A3B-repo/snapshots/*/. /root/Qwen3-30B-A3B && \
            /root/runner_qwen3_30b_a3b.sh 2>&1 | tee /root/logs/qwen3_30b_a3b_convert.out"
