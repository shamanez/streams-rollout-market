#!/usr/bin/env bash
# Inside-container conversion: HF Qwen/Qwen3-32B → Megatron torch-dist.
#
# Mirrors scripts/live/megatron_convert_qwen3_moe.md (the MoE runbook).
# Qwen3-32B is dense — no MoE flags. The slimerl/slime:latest image
# already ships /root/slime/scripts/models/qwen3-32B.sh with the right
# MODEL_ARGS (num_layers=64, hidden_size=5120, num_attention_heads=40,
# num_query_groups=8, kv_channels=128, ffn_hidden_size=27392,
# vocab_size=152064), so this script just sources it.
#
# Host layout (mirrors the MoE recipe):
#   ~/megatron_conversion/hf_model_qwen3_32b/  → mounted at /root/Qwen3-32B/
#   ~/megatron_conversion/qwen3_32b_torch_dist/ → mounted at /root/Qwen3-32B_torch_dist/
#   ~/megatron_conversion/logs/                → mounted at /root/logs/
#
# Launch with the same docker run flags as the MoE runbook
# (--gpus all --ipc=host --shm-size=64g --ulimit memlock=-1
# --ulimit stack=67108864) and bash -lc "/root/runner_qwen3_32b.sh".
set -ex
cd /root/slime
source /root/slime/scripts/models/qwen3-32B.sh

# Atomic save needs an empty target dir.
rm -rf /root/Qwen3-32B_torch_dist/*

PYTHONPATH=/root/Megatron-LM/:/root/slime/ \
torchrun --nproc-per-node 4 \
  /root/slime/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/Qwen3-32B/ \
  --save /root/Qwen3-32B_torch_dist/
