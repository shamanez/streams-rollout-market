#!/usr/bin/env bash
# Launch sglang serving Qwen3-32B with tool-calling enabled, mirroring
# the vLLM serve script's interface. Same port (8000), so the agent
# runner doesn't care which engine is on the other end of the tunnel.
#
# Usage on the spot:
#   MODEL=Qwen/Qwen3-32B   bash ~/serve_qwen_for_agent_sglang.sh
#   MODEL=Qwen/Qwen3-32B-FP8 bash ~/serve_qwen_for_agent_sglang.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
TOOL_PARSER="${TOOL_PARSER:-qwen25}"
DTYPE="${DTYPE:-auto}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-8192}"
MEM_FRAC="${MEM_FRAC:-0.85}"

# sglang needs a CUDA-13 nvcc visible on PATH for FlashInfer JIT compilation.
export CUDA_HOME="${CUDA_HOME:-/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13}"
export PATH="$CUDA_HOME/bin:$PATH"

source ~/sglenv/bin/activate
export HF_HOME=/home/ubuntu/hf-cache

echo "[serve] sglang model=$MODEL parser=$TOOL_PARSER tp=$TP dtype=$DTYPE port=$PORT"
exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp "$TP" \
    --dtype "$DTYPE" \
    --context-length "$MAX_LEN" \
    --mem-fraction-static "$MEM_FRAC" \
    --tool-call-parser "$TOOL_PARSER" \
    --disable-cuda-graph \
    --attention-backend triton \
    --host 0.0.0.0 \
    --port "$PORT"
