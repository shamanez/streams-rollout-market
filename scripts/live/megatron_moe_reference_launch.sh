#!/usr/bin/env bash
# Host-side launcher for the Qwen3-30B-A3B Megatron-bf16 MoE teacher-force
# pass + router trace emission.
#
# Sibling of scripts/live/megatron_reference_launch.sh (which targets
# the dense Qwen3-32B checkpoint). Bind-mounts the MoE torch-dist
# checkpoint under $CKPT_ROOT/qwen3-30b-a3b/megatron into the slimerl/
# slime container as /root/Qwen3-30B-A3B_torch_dist (matching the path
# convention the inside-container runner expects).
#
# Portable: every path is env-driven with $CKPT_ROOT defaulting to
# $HOME/checkpoint. To target a fresh spot, run bootstrap.sh first.
#
# Calls run_megatron_moe_reference.py with --emit-router so the script
# writes BOTH /host_tmp/trainer_megatron-bf16.json (per-response-token
# logprobs) AND /host_tmp/megatron_router.json (top-k expert IDs per
# (response_token, MoE layer), length R).
#
# Usage on the spot host:
#   bash ~/megatron_moe_reference_launch.sh
#
# Inputs (host-side, must exist BEFORE invoking):
#   /tmp/rollout.json  — {model, prompt_token_ids, response_token_ids, ...}
#                        (constructed from /tmp/vllm_router.json after a
#                        vLLM-MoE rollout).
#
# Outputs (host-side, after the container exits):
#   /tmp/trainer_megatron-bf16.json  — trainer-side logprobs (length R)
#   /tmp/megatron_router.json        — Megatron MoE router trace (length R)
set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:-$HOME/checkpoint}"
CKPT_DIR="${CKPT_DIR:-$CKPT_ROOT/qwen3-30b-a3b/megatron}"
HF_MODEL_DIR="${HF_MODEL_DIR:-$(readlink -f "$CKPT_ROOT/qwen3-30b-a3b/hf" 2>/dev/null || echo "$CKPT_ROOT/qwen3-30b-a3b/hf")}"
ROLLOUT_FILE_HOST="${ROLLOUT_FILE_HOST:-/tmp/rollout.json}"
RUNNER_PY="${RUNNER_PY:-$(dirname "$(readlink -f "$0")")/run_megatron_moe_reference.py}"
RUNNER_SH="${RUNNER_SH:-$(dirname "$(readlink -f "$0")")/megatron_qwen3_32b_dense_runner.sh}"
IMAGE="${IMAGE:-slimerl/slime:latest}"

if [[ ! -d "$CKPT_DIR/release" ]]; then
  echo "ERROR: CKPT_DIR=$CKPT_DIR has no release/ subdir " \
       "(was the Qwen3-30B-A3B HF→Megatron conversion completed?)" >&2
  exit 2
fi
if [[ ! -f "$HF_MODEL_DIR/config.json" ]]; then
  echo "ERROR: HF_MODEL_DIR=$HF_MODEL_DIR has no config.json " \
       "(slime's set_default_megatron_args needs the HF model dir to " \
       "construct the tokenizer)" >&2
  exit 2
fi
if [[ ! -f "$ROLLOUT_FILE_HOST" ]]; then
  echo "ERROR: ROLLOUT_FILE_HOST=$ROLLOUT_FILE_HOST not found " \
       "(run run_vllm_moe_rollout first and construct rollout.json)" >&2
  exit 2
fi
if [[ ! -f "$RUNNER_PY" ]]; then
  echo "ERROR: RUNNER_PY=$RUNNER_PY not found" >&2
  exit 2
fi
if [[ ! -f "$RUNNER_SH" ]]; then
  echo "ERROR: RUNNER_SH=$RUNNER_SH not found " \
       "(the inside-container wrapper that sources qwen3-30B-A3B MODEL_ARGS)" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
chmod -R a+rwx "$WORK_DIR"
# Detect single-dict vs multi-rollout list and name the file the
# inside-container runner expects (rollouts.json for lists,
# rollout.json for single dicts — see run_megatron_moe_reference.py
# --rollouts-path / --rollouts-path-fallback).
if python3 -c "import json,sys;d=json.load(open(sys.argv[1]));sys.exit(0 if isinstance(d,list) else 1)" "$ROLLOUT_FILE_HOST"; then
  cp "$ROLLOUT_FILE_HOST" "$WORK_DIR/rollouts.json"
  echo "[launch] input is a list — copied as WORK_DIR/rollouts.json"
else
  cp "$ROLLOUT_FILE_HOST" "$WORK_DIR/rollout.json"
  echo "[launch] input is a single dict — copied as WORK_DIR/rollout.json"
fi

# The HF snapshot dir is full of relative symlinks pointing to
# `../../blobs/<hash>` outside the mount, so a naive bind-mount makes
# `tokenizer.json` etc. unreadable inside the container. Dereference
# into a flat directory and bind-mount that instead.
HF_FLAT="$(mktemp -d)"
chmod -R a+rwx "$HF_FLAT"
cp -rL "$HF_MODEL_DIR"/. "$HF_FLAT"/

echo "[launch] CKPT_DIR=$CKPT_DIR"
echo "[launch] ROLLOUT_FILE_HOST=$ROLLOUT_FILE_HOST"
echo "[launch] WORK_DIR=$WORK_DIR (mounted as /host_tmp)"
echo "[launch] HF_MODEL_DIR=$HF_MODEL_DIR (dereferenced to $HF_FLAT)"
echo "[launch] RUNNER_PY=$RUNNER_PY"
echo "[launch] RUNNER_SH=$RUNNER_SH"

# Clear stale outputs the previous run may have left as root-owned in
# /tmp; otherwise the post-docker cp can't overwrite them.
rm -f /tmp/megatron_router.json /tmp/trainer_megatron-bf16.json 2>/dev/null || \
  sudo rm -f /tmp/megatron_router.json /tmp/trainer_megatron-bf16.json 2>/dev/null || true

docker run --rm \
  --gpus all --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$CKPT_DIR":/root/Qwen3-30B-A3B_torch_dist:ro \
  -v "$HF_FLAT":/root/Qwen3-30B-A3B:ro \
  -v "$RUNNER_PY":/root/run_megatron_reference.py:ro \
  -v "$RUNNER_SH":/root/runner.sh:ro \
  -v "$WORK_DIR":/host_tmp \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  "$IMAGE" \
  bash -lc "/root/runner.sh && \
    chown \"\$HOST_UID:\$HOST_GID\" /host_tmp/*.json"

# Optional per-run suffix so multiple consecutive launches (e.g.
# bf16 then fp8 rollouts) don't trample each other's outputs. Set
# OUT_SUFFIX=-bf16_rollouts to get /tmp/trainer_megatron-bf16-bf16_rollouts.json etc.
OUT_SUFFIX="${OUT_SUFFIX:-}"

# Single-rollout mode writes trainer_megatron-bf16.json; multi-mode
# writes BOTH trainer_megatron-bf16.json (first entry) AND
# trainers_megatron-bf16.json (full list). Mirror both.
if [[ -f "$WORK_DIR/trainer_megatron-bf16.json" ]]; then
  cp "$WORK_DIR/trainer_megatron-bf16.json" "/tmp/trainer_megatron-bf16${OUT_SUFFIX}.json"
  echo "[launch] wrote /tmp/trainer_megatron-bf16${OUT_SUFFIX}.json"
fi
if [[ -f "$WORK_DIR/trainers_megatron-bf16.json" ]]; then
  cp "$WORK_DIR/trainers_megatron-bf16.json" "/tmp/trainers_megatron-bf16${OUT_SUFFIX}.json"
  echo "[launch] wrote /tmp/trainers_megatron-bf16${OUT_SUFFIX}.json"
fi
if [[ -f "$WORK_DIR/megatron_router.json" ]]; then
  cp "$WORK_DIR/megatron_router.json" "/tmp/megatron_router${OUT_SUFFIX}.json"
  echo "[launch] wrote /tmp/megatron_router${OUT_SUFFIX}.json"
fi
