#!/usr/bin/env bash
# Inside-container conversion command. Sourced via docker bash -lc.
set -ex
cd /root/slime
source /root/slime/scripts/models/qwen3-30B-A3B.sh

# Cleanup any prior incomplete run so torch-dist's atomic save can claim the dir.
rm -rf /root/Qwen3-30B-A3B_torch_dist/*

PYTHONPATH=/root/Megatron-LM/:/root/slime/ \
torchrun --nproc-per-node 4 \
  /root/slime/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen3-30B-A3B/ \
  --save /root/Qwen3-30B-A3B_torch_dist/
