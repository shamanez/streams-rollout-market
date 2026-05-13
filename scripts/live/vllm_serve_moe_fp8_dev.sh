#!/usr/bin/env bash
# Serve Qwen3-30B-A3B-FP8 (MoE, FP8 quant) on the spot with router-replay
# flags enabled.
#
# Thin sibling of ``vllm_serve_moe_dev.sh``: identical flags + venv,
# only the MODEL and LOG defaults change. Both scripts source the same
# patched dev venv (~/rmenv-dev/) which carries the routed_experts
# HTTP shim. FP8 routing traces flow through the same proxy path and
# land in /tmp/hermes_probes.jsonl with the same shape as bf16
# (``[gen_tokens, 48, 8]`` per assistant turn).
#
# Run on the SPOT in a tmux pane (or via nohup+setsid):
#   bash ~/streams-rollout-market/scripts/live/vllm_serve_moe_fp8_dev.sh
#
# Env (defaults — same as bf16 except MODEL and LOG):
#   MODEL        Qwen/Qwen3-30B-A3B-FP8
#   LOG          $HOME/vllm-logs/moe_fp8_serve.log
#   (HOST, PORT, TP_SIZE, MAX_LEN, GPU_UTIL, RMENV_DEV, HF_HOME inherit
#    the same defaults documented in ``vllm_serve_moe_dev.sh``.)

set -euo pipefail

export MODEL="${MODEL:-Qwen/Qwen3-30B-A3B-FP8}"
export LOG="${LOG:-$HOME/vllm-logs/moe_fp8_serve.log}"

exec bash "$(dirname "$(readlink -f "$0")")/vllm_serve_moe_dev.sh"
