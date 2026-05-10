#!/usr/bin/env bash
# Launch vLLM serving Qwen3-32B (or its FP8 variant) with tool-calling
# enabled, on the spot host. Runs in the foreground; kill with Ctrl-C
# or `pkill -f "vllm serve"` from the spot.
#
# Usage on the spot:
#   MODEL=Qwen/Qwen3-32B   bash ~/serve_qwen_for_agent.sh
#   MODEL=Qwen/Qwen3-32B-FP8 bash ~/serve_qwen_for_agent.sh
#
# Bind to 0.0.0.0:8000 — connect from the Mac via SSH tunnel:
#   ssh -L 8000:localhost:8000 my-vllm-spot-instance

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B}"
PORT="${PORT:-8000}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"
DTYPE="${DTYPE:-auto}"
TP="${TP:-4}"
MAX_LEN="${MAX_LEN:-8192}"
MEM_FRAC="${MEM_FRAC:-0.85}"

source ~/rmenv/bin/activate
export HF_HOME=/home/ubuntu/hf-cache

echo "[serve] model=$MODEL parser=$TOOL_PARSER tp=$TP dtype=$DTYPE port=$PORT"
exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP" \
    --dtype "$DTYPE" \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$MEM_FRAC" \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --host 0.0.0.0 \
    --port "$PORT"
