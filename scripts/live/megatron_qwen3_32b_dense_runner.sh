#!/usr/bin/env bash
# Inside-container wrapper for run_megatron_reference.py.
#
# Sources the Qwen3-30B-A3B MODEL_ARGS that slime ships with, then
# runs the reference script under torchrun with TP=4 (matching the
# torch-dist checkpoint that was saved at TP=4 / PP=1).
set -ex

# Megatron requires this when TP > 1 (asserts on rank 0..N-1 otherwise).
export CUDA_DEVICE_MAX_CONNECTIONS=1

cd /root/slime
# Pull in MODEL_ARGS=( ... ) for Qwen3-30B-A3B (48 layers, 128 experts).
source /root/slime/scripts/models/qwen3-30B-A3B.sh

# Args that load_checkpoint + slime init need.
LOAD_ARGS=(
    --load /root/Qwen3-30B-A3B_torch_dist
    --tensor-model-parallel-size 4
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --micro-batch-size 1
    --global-batch-size 4
    --no-load-optim
    --no-load-rng
    --bf16
    --use-checkpoint-args
)

PYTHONPATH=/root/Megatron-LM/:/root/slime/ \
torchrun --nproc-per-node 4 \
    /root/run_megatron_reference.py \
    "${MODEL_ARGS[@]}" \
    "${LOAD_ARGS[@]}" \
    --emit-router
