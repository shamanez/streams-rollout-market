#!/usr/bin/env bash
# Serve Qwen3-30B-A3B (MoE) on the spot with router-replay flags enabled.
#
# This is the MoE counterpart of CLAUDE.md "End-to-end reproduction" step 1.
# The dev variant sources ``~/rmenv-dev/`` (patched to emit
# ``routed_experts`` over the OpenAI HTTP shim — see
# ``scripts/live/patch_vllm_routed_experts_http.py``) instead of the
# baseline ``~/rmenv/``, so the proxy's sidecar JSONL captures router
# traces alongside logprobs.
#
# Run on the SPOT in a tmux pane:
#   bash ~/streams-rollout-market/scripts/live/vllm_serve_moe_dev.sh
#
# Env (defaults):
#   MODEL        Qwen/Qwen3-30B-A3B
#   HOST         0.0.0.0
#   PORT         8000
#   TP_SIZE      4
#   MAX_LEN      40960
#   GPU_UTIL     0.85
#   LOG          $HOME/vllm-logs/moe_bf16_serve.log
#   RMENV_DEV    $HOME/rmenv-dev          (the patched dev venv)
#   HF_HOME      $HOME/hf-cache

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-30B-A3B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP_SIZE="${TP_SIZE:-4}"
MAX_LEN="${MAX_LEN:-40960}"
GPU_UTIL="${GPU_UTIL:-0.85}"
LOG="${LOG:-$HOME/vllm-logs/moe_bf16_serve.log}"
RMENV_DEV="${RMENV_DEV:-$HOME/rmenv-dev}"
export HF_HOME="${HF_HOME:-$HOME/hf-cache}"

mkdir -p "$(dirname "$LOG")"

if [[ ! -x "$RMENV_DEV/bin/python" ]]; then
  echo "vllm_serve_moe_dev: $RMENV_DEV not a venv (no bin/python)." >&2
  echo "Bootstrap it first per .claude/evidence/iter01_spot_vllm_dev/iter01.md." >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$RMENV_DEV/bin/activate"

# Sanity: the routed-experts HTTP patch must already be applied. If not,
# refuse to serve — otherwise the proxy will silently capture nothing.
python - <<'PY'
import sys
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionResponseChoice
from vllm.entrypoints.openai.completion.protocol import CompletionResponseChoice
missing = [
    cls.__name__
    for cls in (CompletionResponseChoice, ChatCompletionResponseChoice)
    if "routed_experts" not in cls.model_fields
]
if missing:
    print(f"FATAL: routed_experts field missing on: {missing}", file=sys.stderr)
    print("Apply scripts/live/patch_vllm_routed_experts_http.py first.", file=sys.stderr)
    sys.exit(3)
PY

echo "vllm_serve_moe_dev: serving $MODEL on $HOST:$PORT (TP=$TP_SIZE, max_len=$MAX_LEN)" \
  | tee "$LOG"

# ``--no-async-scheduling`` is required: routed_experts capture is
# incompatible with async scheduling (per the upstream routed-experts
# replay docs at commit a7b801e).
#
# ``--enable-expert-parallel`` keeps the per-token routing trace
# distinct across the 4 L40S TP ranks; without it, the trace gathers on
# TP rank 0 only and the shape would still be correct but the path is
# untested at this commit.
exec vllm serve "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --dtype auto \
    --max-model-len "$MAX_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-return-routed-experts \
    --enable-expert-parallel \
    --no-async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --host "$HOST" \
    --port "$PORT" \
    >> "$LOG" 2>&1
