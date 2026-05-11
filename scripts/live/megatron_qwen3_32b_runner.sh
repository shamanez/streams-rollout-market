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
# Bind-mount layout (host → container), set by
# scripts/live/megatron_qwen3_32b_launch.sh:
#   ~/hf-cache/hub/models--Qwen--Qwen3-32B/  → /root/Qwen3-32B-repo/ (ro)
#   ~/megatron_conversion/qwen3_32b_torch_dist/ → /root/Qwen3-32B_torch_dist/
#   ~/megatron_conversion/logs/                → /root/logs/
#
# Mounting the *repo* (not just the snapshot) is load-bearing: the
# snapshot directory contains relative `../../blobs/...` symlinks; only
# the repo-level mount makes those resolve inside the container. The
# previous staging layout (host-absolute symlinks into ~/hf-cache) left
# tokenizer.json dangling and the HF tokenizer init crashed before any
# weight conversion could happen.
set -ex
cd /root/slime
source /root/slime/scripts/models/qwen3-32B.sh

# Pick the snapshot directory from the bind-mounted repo. If
# HF_CHECKPOINT is set by the caller we honour it (back-compat with
# the legacy host layout that staged files at /root/Qwen3-32B/).
if [[ -z "${HF_CHECKPOINT:-}" ]]; then
  HF_CHECKPOINT="$(ls -d /root/Qwen3-32B-repo/snapshots/*/ 2>/dev/null | head -1)"
fi
if [[ -z "${HF_CHECKPOINT}" || ! -f "${HF_CHECKPOINT%/}/config.json" ]]; then
  echo "ERROR: HF_CHECKPOINT (${HF_CHECKPOINT}) is not a valid HF snapshot directory" >&2
  exit 2
fi
echo "HF_CHECKPOINT=${HF_CHECKPOINT}"

# Atomic save needs an empty target dir.
rm -rf /root/Qwen3-32B_torch_dist/*

PYTHONPATH=/root/Megatron-LM/:/root/slime/ \
torchrun --nproc-per-node 4 \
  /root/slime/tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save /root/Qwen3-32B_torch_dist/
